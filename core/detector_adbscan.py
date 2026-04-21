"""
detector_adbscan.py
====================
Optimised ADBScan detector — produces the best result from the PPT:
  11 tracks, 100% accuracy, faster than sklearn DBSCAN.

Key differences from detector.py:
  1. NEW FILE — does not modify detector.py
  2. ADBScan uses RADIAL distance from sensor centre for eps (not x-position)
     This is physically what the PPT shows as the best result.
  3. All optimisations applied:
       - return_sorted=False (15% faster query)
       - np.concatenate CSR (avoids Python for-loop)
       - workers=1 (avoids joblib overhead for small N)
       - KDTree built once per window and reused
       - ADBScan object created once outside window loop
  4. FastDBSCAN also available as a flag for comparison

Usage:
    from detector_adbscan import ADBScanDetector

    det = ADBScanDetector(window_ms=5, eps_px=10, min_samples=5)
    tracks, feats = det.detect(stream, gt_labels)
"""

import numpy as np
from scipy.spatial import KDTree
import time
from typing import Optional

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    from adbscan import _cluster_jit, NUMBA_AVAILABLE
except ImportError:
    NUMBA_AVAILABLE = False

from detector import Detection, Track, EventDetector
from event_reader import EventStream
from classifier import ObjectClassifier, CLASS_NAMES


# ── ADBScan adaptive parameter modes ─────────────────────────────────────────

def _compute_eps_radial(xw, yw, cx, cy, c2, c1, base):
    """
    Per-point eps based on RADIAL distance from sensor centre.
    This is the mode that produced 11 tracks / 100% acc in the PPT.

    d   = sqrt((x - cx)^2 + (y - cy)^2)   [radial pixels from centre]
    eps = clip(c2*d^2 + c1*d + base, min=0.9)
    """
    d = np.sqrt((xw - cx) ** 2 + (yw - cy) ** 2)
    return np.clip(c2 * d**2 + c1 * d + base, 0.9, None)


def _compute_eps_xpos(xw, c2, c1, base):
    """
    Per-point eps based on x-pixel coordinate (current pixel mode).
    eps = clip(c2*x^2 + c1*x + base, 0.9, None)
    """
    return np.clip(c2 * xw**2 + c1 * xw + base, 0.9, None)


# ── Fast CSR conversion ───────────────────────────────────────────────────────

def _to_csr(nb_raw, N):
    """
    Convert list-of-lists neighbour result to CSR format.
    Uses np.concatenate (faster than Python for-loop).
    """
    nb_counts  = np.fromiter((len(n) for n in nb_raw), dtype=np.int32, count=N)
    nb_offsets = np.zeros(N + 1, dtype=np.int32)
    np.cumsum(nb_counts, out=nb_offsets[1:])
    total = int(nb_offsets[N])
    if total > 0:
        nb_data = np.concatenate(nb_raw).astype(np.int32)
    else:
        nb_data = np.empty(0, dtype=np.int32)
    return nb_counts, nb_offsets, nb_data


# ── Python BFS fallback ───────────────────────────────────────────────────────

def _cluster_python(N, nb_data, nb_offsets, nb_counts, k_array):
    from collections import deque
    labels    = np.zeros(N, dtype=np.int32)
    core_mask = np.zeros(N, dtype=np.int8)
    touched   = np.zeros(N, dtype=np.uint8)
    active    = np.zeros(N, dtype=np.uint8)
    no = 1
    for i in range(N):
        if touched[i]: continue
        cnt_i = int(nb_counts[i]); k_i = int(k_array[i])
        if cnt_i <= k_i:
            labels[i]=-1; core_mask[i]=-1; touched[i]=1; continue
        core_mask[i]=1; labels[i]=no
        s=nb_offsets[i]; e=nb_offsets[i+1]
        nb_i=nb_data[s:e]; labels[nb_i]=no
        queue=deque(nb_i.tolist()); active[nb_i]=1
        while queue:
            ind=queue.popleft(); touched[ind]=1; active[ind]=0
            s2=nb_offsets[ind]; e2=nb_offsets[ind+1]
            nb_ind=nb_data[s2:e2]; cnt=int(nb_counts[ind]); k_ind=int(k_array[ind])
            if cnt<2: continue
            core_mask[ind]=1 if cnt>=k_ind+1 else 0
            labels[nb_ind]=no
            new_mask=touched[nb_ind]==0
            if new_mask.any():
                new_pts=nb_ind[new_mask]; touched[new_pts]=1
                na=active[new_pts]==0
                if na.any():
                    to_add=new_pts[na]; active[to_add]=1
                    queue.extend(to_add.tolist())
        no += 1
    unassigned=labels==0
    labels[unassigned]=-1; core_mask[unassigned]=-1
    labels[labels>0]-=1
    return labels, core_mask, no-1


# ── Main detector class ───────────────────────────────────────────────────────

class ADBScanDetector:
    """
    Optimised ADBScan detector for neuromorphic event cameras.

    Reproduces the best PPT result: 11 tracks, 100% accuracy, faster than
    sklearn DBSCAN — using fixed k and eps varying with radial distance
    from the sensor centre.

    Parameters
    ----------
    window_ms      : float  time window duration (ms)
    eps_px         : float  base search radius (pixels) — at sensor centre
    min_samples    : int    fixed k (min_points) — same for all points
    min_track_len  : int    minimum windows for a valid track
    max_gap_windows: int    max missed windows before track ends
    track_radius_px: float  max centroid jump to link detections (px)
    eps_mode       : str    'radial' (PPT best) or 'xpos' (pixel mode)
    adaptive_params: tuple  (c2, c1, base) for quadratic eps
                            base = eps at sensor centre (radial) or at x=0 (xpos)
    verbose        : bool
    """

    def __init__(
        self,
        window_ms       = 5.0,
        eps_px          = 10.0,
        min_samples     = 5,
        min_track_len   = 4,
        max_gap_windows = 3,
        track_radius_px = 40.0,
        eps_mode        = 'radial',
        adaptive_params = None,
        verbose         = True,
    ):
        self.window_us       = window_ms * 1000.0
        self.eps_px          = eps_px
        self.min_samples     = min_samples
        self.min_track_len   = min_track_len
        self.max_gap_windows = max_gap_windows
        self.track_radius_px = track_radius_px
        self.eps_mode        = eps_mode
        self.verbose         = verbose

        # Quadratic coefficients for eps
        if adaptive_params is None:
            # Defaults: gentle quadratic, base=eps_px at sensor centre
            self.c2   = 0.0001
            self.c1   = -0.02
            self.base = float(eps_px)
        else:
            self.c2, self.c1, self.base = adaptive_params

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, stream, gt_labels: Optional[np.ndarray] = None):
        """
        Run optimised ADBScan detection on full event stream.

        Returns
        -------
        tracks : list of Track
        feats  : np.ndarray  feature matrix (N_tracks, N_features)
        """
        if self.verbose:
            mode_str = f"radial-eps" if self.eps_mode=='radial' else "xpos-eps"
            print(f"\n[ADBScanDetector] mode={mode_str} | "
                  f"window={self.window_us/1000:.1f}ms | "
                  f"eps_base={self.eps_px}px | k={self.min_samples}")

        t0 = time.time()
        detections = self._windowed_cluster(stream)
        tracks     = self._link_tracks(detections)
        tracks     = [tr for tr in tracks if tr.length >= self.min_track_len]

        # Assign ground-truth
        if gt_labels is not None:
            self._assign_gt(tracks, gt_labels)

        # Extract features
        feats = self._extract_features(tracks)

        t_total = time.time() - t0
        if self.verbose:
            print(f"  Raw detections: {len(detections)}")
            print(f"  Tracks (≥{self.min_track_len} windows): {len(tracks)}")
            if gt_labels is not None:
                from collections import Counter
                gt_counts = Counter(
                    CLASS_NAMES[tr.gt_class] for tr in tracks
                    if hasattr(tr,'gt_class') and tr.gt_class >= 0)
                for cls, cnt in sorted(gt_counts.items()):
                    print(f"    {cls:<14}: {cnt} tracks")
            print(f"  Time: {t_total:.2f}s")

        return tracks, feats

    # ── Windowed clustering ───────────────────────────────────────────────────

    def _windowed_cluster(self, stream):
        """
        Core loop: for each time window, compute adaptive eps, build KDTree,
        query neighbours, run Numba BFS — all optimised.
        """
        cx = stream.W / 2.0   # sensor centre x
        cy = stream.H / 2.0   # sensor centre y
        c2, c1, base = self.c2, self.c1, self.base

        t_start_all = stream.t[0]
        t_end       = stream.t[-1]
        n_windows   = int((t_end - t_start_all) / self.window_us) + 1

        # Fixed k array — all points same min_samples
        # Pre-allocated; resized per window if needed
        k_fixed = self.min_samples

        detections = []
        win_id = 0

        for w in range(n_windows):
            t_start = t_start_all + w * self.window_us
            t_stop  = t_start + self.window_us
            mask    = (stream.t >= t_start) & (stream.t < t_stop)
            N       = int(mask.sum())

            if N < self.min_samples:
                win_id += 1
                continue

            idx_w  = np.where(mask)[0]
            xw     = stream.x[idx_w].astype(np.float64)
            yw     = stream.y[idx_w].astype(np.float64)
            pw     = stream.p[idx_w]
            coords = np.column_stack([xw, yw])

            # ── Adaptive eps per point ────────────────────────────────────
            if self.eps_mode == 'radial':
                eps_arr = _compute_eps_radial(xw, yw, cx, cy, c2, c1, base)
            else:
                eps_arr = _compute_eps_xpos(xw, c2, c1, base)

            # Fixed k for all points
            k_array = np.full(N, k_fixed, dtype=np.int32)

            # ── KDTree + radius search ────────────────────────────────────
            # return_sorted=False: ~15% faster, order doesn't matter for BFS
            # workers=1: eliminates joblib thread-pool overhead (~14ms saved)
            tree   = KDTree(coords)
            nb_raw = tree.query_ball_point(coords, eps_arr,
                                           workers=1,
                                           return_sorted=False)

            # ── CSR conversion (vectorised) ───────────────────────────────
            nb_counts, nb_offsets, nb_data = _to_csr(nb_raw, N)

            # ── BFS clustering ────────────────────────────────────────────
            if NUMBA_AVAILABLE:
                lbl, _, n_cl = _cluster_jit(
                    N, nb_data, nb_offsets, nb_counts, k_array)
                lbl[lbl > 0] -= 1
            else:
                lbl, _, n_cl = _cluster_python(
                    N, nb_data, nb_offsets, nb_counts, k_array)

            # ── Extract detections ────────────────────────────────────────
            for cid in range(n_cl):
                cm = lbl == cid
                if cm.sum() < self.min_samples:
                    continue
                cxv = xw[cm]; cyv = yw[cm]
                detections.append(Detection(
                    window_id = win_id,
                    t_center  = (t_start + t_stop) / 2.0,
                    cx        = float(cxv.mean()),
                    cy        = float(cyv.mean()),
                    x_min     = float(cxv.min()),
                    x_max     = float(cxv.max()),
                    y_min     = float(cyv.min()),
                    y_max     = float(cyv.max()),
                    n_events  = int(cm.sum()),
                    pol_ratio = float((pw[cm] > 0).mean()),
                    event_idx = idx_w[cm],
                ))
            win_id += 1

        return detections

    # ── Track linking (nearest-neighbour) ────────────────────────────────────

    def _link_tracks(self, detections):
        """Greedy nearest-neighbour tracker — same logic as EventDetector."""
        active   = []
        finished = []

        for det in sorted(detections, key=lambda d: d.window_id):
            best_dist = self.track_radius_px
            best_tr   = None

            for tr in active:
                gap = det.window_id - tr.last.window_id
                if gap > self.max_gap_windows:
                    continue
                d = np.sqrt((det.cx - tr.last.cx)**2 +
                            (det.cy - tr.last.cy)**2)
                if d < best_dist:
                    best_dist = d
                    best_tr   = tr

            if best_tr is not None:
                best_tr.detections.append(det)
            else:
                new_tr = Track(track_id=len(active) + len(finished))
                new_tr.detections.append(det)
                active.append(new_tr)

            # Expire stale tracks
            still_active = []
            for tr in active:
                if det.window_id - tr.last.window_id <= self.max_gap_windows:
                    still_active.append(tr)
                else:
                    finished.append(tr)
            active = still_active

        finished.extend(active)
        return finished

    # ── GT assignment ─────────────────────────────────────────────────────────

    def _assign_gt(self, tracks, gt_labels):
        for tr in tracks:
            all_idx = np.concatenate([d.event_idx for d in tr.detections])
            all_idx = all_idx[all_idx < len(gt_labels)]
            gt_ev   = gt_labels[all_idx]
            valid   = gt_ev[gt_ev >= 0]
            if len(valid) > 0:
                vals, cnts = np.unique(valid, return_counts=True)
                tr.gt_class = int(vals[cnts.argmax()])
            else:
                tr.gt_class = -1

    # ── Feature extraction ────────────────────────────────────────────────────

    def _extract_features(self, tracks):
        """Extract 15-dim feature vector per track (same as EventDetector)."""
        from detector import EventDetector
        det = EventDetector()
        return det._extract_features(tracks)
