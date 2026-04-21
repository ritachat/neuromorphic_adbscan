"""
detector_3d.py
==============
ADBScan in 3D (x, y, t) — spatiotemporal clustering for neuromorphic events.

WHY 3D?
--------
Standard 2D clustering works in (x,y) per time window. The temporal dimension
carries extra discriminative information:

  Rain streaks   : narrow in x (1-5px), thin in t (individual drops brief)
                   In 3D space: short thin tubes along the t axis
  Objects        : wide in x,y (Car=55px, Truck=80px), persistent across t
                   In 3D space: thick blobs spanning full window duration
  Snow blobs     : small in x,y, brief in t
                   In 3D space: tiny spheroids

TIME SCALING
------------
Raw timestamps are in microseconds (0–5000 for a 5ms window) while x,y are
in pixels (0–346/260). Direct clustering would make t dominate. We scale t so
that 1 "time pixel" ≈ 1 spatial pixel:

    t_scaled = (t - t_window_start) / T_SCALE_US_PER_PX

Calibrated so that an object moving at ~3.5px/ms traverses 1px in ~286us,
giving T_SCALE ≈ 286. Tuned empirically to T_SCALE=200 (best cluster recall).

ADAPTIVE ε IN 3D
-----------------
ε is adapted based on radial distance from the sensor centre in (x,y) only —
the temporal axis uses the same eps since it has been normalised to px units.

    d   = sqrt((x-cx)² + (y-cy)²)
    ε   = clip(c2·d² + c1·d + base, 0.9)   [same as 2D ADBScan]
    coords_3d = [x, y, t_scaled]

The same eps applies in all 3 dimensions after scaling.

3D FEATURE EXTRACTION
----------------------
After clustering, tracks carry richer spatiotemporal features:
    t_span_ms    : temporal extent of the cluster (ms)
    vx, vy       : mean velocity (px/ms) from linear regression on (x,t) / (y,t)
    aspect_3d    : ratio of spatial spread to temporal spread
    compactness  : event density in 3D bounding volume

These additional features are fed to the weather filter for improved separation.

Usage:
    from detector_3d import ADBScan3DDetector

    det = ADBScan3DDetector(window_ms=5, eps_px=10, t_scale_us=200)
    tracks, feats = det.detect(stream, gt_labels)
"""

import numpy as np
from scipy.spatial import KDTree
import time
from typing import Optional

from adbscan       import _cluster_jit, NUMBA_AVAILABLE
from detector      import Detection, Track, EventDetector
from classifier    import CLASS_NAMES
from detector_adbscan import _compute_eps_radial, _to_csr, _cluster_python

# ── Default parameters ────────────────────────────────────────────────────────
DEFAULT_T_SCALE = 200     # us per spatial pixel — converts t to px-equivalent
FEATURE_NAMES_3D = [
    'cx', 'cy', 'width', 'height', 'aspect',
    'n_events', 'pol_ratio', 'speed_x', 'speed_y', 'speed',
    'straightness', 'spread_x', 'spread_y',
    # 3D-specific extras
    't_span_ms', 'ev_density_3d',
]


class ADBScan3DDetector:
    """
    3D ADBScan detector: clusters events in (x, y, t) space.

    Compared to 2D ADBScan:
    - Adaptive ε based on radial (x,y) distance (same as PPT best)
    - t axis scaled to px units via t_scale_us parameter
    - Same Numba JIT BFS — unchanged, works for any dimensionality
    - Adds 3D-specific features to each detection

    Parameters
    ----------
    window_ms      : float   time window (ms)
    eps_px         : float   base search radius (pixels) at sensor centre
    min_samples    : int     k — min neighbours to form a core point
    min_track_len  : int     min windows a track must span
    t_scale_us     : float   microseconds per spatial pixel for t normalisation
                             200 = good balance (object motion ≈ 3.5px/ms)
    adaptive_params: tuple   (c2, c1, base) for quadratic eps(d)
    verbose        : bool
    """

    def __init__(
        self,
        window_ms      = 5.0,
        eps_px         = 10.0,
        min_samples    = 5,
        min_track_len  = 4,
        max_gap_windows= 3,
        track_radius_px= 40.0,
        t_scale_us     = DEFAULT_T_SCALE,
        adaptive_params= None,
        verbose        = True,
    ):
        self.window_us       = window_ms * 1000.0
        self.eps_px          = eps_px
        self.min_samples     = min_samples
        self.min_track_len   = min_track_len
        self.max_gap_windows = max_gap_windows
        self.track_radius_px = track_radius_px
        self.t_scale_us      = t_scale_us
        self.verbose         = verbose

        if adaptive_params is None:
            self.c2, self.c1, self.base = 0.0001, -0.02, float(eps_px)
        else:
            self.c2, self.c1, self.base = adaptive_params

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, stream, gt_labels=None):
        if self.verbose:
            print(f"\n[ADBScan3DDetector] window={self.window_us/1000:.1f}ms | "
                  f"eps={self.eps_px}px | k={self.min_samples} | "
                  f"t_scale={self.t_scale_us}us/px")

        t0 = time.time()
        detections = self._windowed_cluster(stream)
        tracks     = self._link_tracks(detections)
        tracks     = [tr for tr in tracks if tr.length >= self.min_track_len]

        if gt_labels is not None:
            self._assign_gt(tracks, gt_labels)

        feats = self._extract_features_3d(tracks, stream)

        if self.verbose:
            print(f"  Raw detections: {len(detections)}")
            print(f"  Tracks (≥{self.min_track_len}): {len(tracks)}")
            if gt_labels is not None:
                from collections import Counter
                gc = Counter(CLASS_NAMES[tr.gt_class]
                             for tr in tracks
                             if hasattr(tr,'gt_class') and 0<=tr.gt_class<len(CLASS_NAMES))
                for cls,cnt in sorted(gc.items()):
                    print(f"    {cls:<14}: {cnt}")
            print(f"  Time: {time.time()-t0:.2f}s")

        return tracks, feats

    # ── Core 3D windowed clustering ───────────────────────────────────────────

    def _windowed_cluster(self, stream):
        cx = stream.W / 2.0
        cy = stream.H / 2.0
        c2, c1, base = self.c2, self.c1, self.base

        t0_stream   = stream.t[0]
        n_windows   = int((stream.t[-1] - t0_stream) / self.window_us) + 1
        k_fixed     = self.min_samples
        detections  = []
        win_id      = 0

        for w in range(n_windows):
            t_start = t0_stream + w * self.window_us
            t_stop  = t_start + self.window_us
            mask    = (stream.t >= t_start) & (stream.t < t_stop)
            N       = int(mask.sum())
            if N < self.min_samples:
                win_id += 1
                continue

            idx_w  = np.where(mask)[0]
            xw     = stream.x[idx_w].astype(np.float64)
            yw     = stream.y[idx_w].astype(np.float64)
            tw     = stream.t[idx_w].astype(np.float64)
            pw     = stream.p[idx_w]

            # Scale t to spatial pixel units
            t_scaled = (tw - t_start) / self.t_scale_us

            # 3D coordinates: [x, y, t_scaled]
            coords3 = np.column_stack([xw, yw, t_scaled])

            # Adaptive eps based on (x,y) radial distance only
            eps_arr = _compute_eps_radial(xw, yw, cx, cy, c2, c1, base)
            k_array = np.full(N, k_fixed, dtype=np.int32)

            # KDTree in 3D + adaptive radius query
            tree   = KDTree(coords3)
            nb_raw = tree.query_ball_point(coords3, eps_arr,
                                           workers=1, return_sorted=False)

            # CSR + BFS
            nb_counts, nb_offsets, nb_data = _to_csr(nb_raw, N)

            if NUMBA_AVAILABLE:
                lbl, _, n_cl = _cluster_jit(N, nb_data, nb_offsets,
                                            nb_counts, k_array)
                lbl[lbl > 0] -= 1
            else:
                lbl, _, n_cl = _cluster_python(N, nb_data, nb_offsets,
                                               nb_counts, k_array)

            # Extract detections with 3D metadata
            for cid in range(n_cl):
                cm = lbl == cid
                if cm.sum() < self.min_samples:
                    continue
                cxv = xw[cm]; cyv = yw[cm]; twv = tw[cm]
                det = Detection(
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
                )
                # Attach 3D metadata directly to detection
                det.t_min_us = float(twv.min())
                det.t_max_us = float(twv.max())
                det.t_span_ms= float((twv.max() - twv.min()) / 1000.0)
                detections.append(det)

            win_id += 1

        return detections

    # ── Track linking (same NN approach as 2D) ────────────────────────────────

    def _link_tracks(self, detections):
        active   = []
        finished = []
        for det in sorted(detections, key=lambda d: d.window_id):
            best_d  = self.track_radius_px
            best_tr = None
            for tr in active:
                gap = det.window_id - tr.last.window_id
                if gap > self.max_gap_windows:
                    continue
                d = np.sqrt((det.cx-tr.last.cx)**2 + (det.cy-tr.last.cy)**2)
                if d < best_d:
                    best_d = d; best_tr = tr
            if best_tr is not None:
                best_tr.detections.append(det)
            else:
                nt = Track(track_id=len(active)+len(finished))
                nt.detections.append(det)
                active.append(nt)
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
                vals,cnts = np.unique(valid, return_counts=True)
                tr.gt_class = int(vals[cnts.argmax()])
            else:
                tr.gt_class = -1

    # ── 3D feature extraction ─────────────────────────────────────────────────

    def _extract_features_3d(self, tracks, stream):
        """
        Extract 15-dim feature vector per track — compatible with 2D features
        PLUS 3D-specific additions.
        """
        # Use base 2D features from EventDetector
        det2d = EventDetector()
        feats_2d = det2d._extract_features(tracks)

        if len(tracks) == 0 or feats_2d is None or len(feats_2d) == 0:
            return feats_2d

        # Add 3D-specific features (appended as extra columns)
        extras = []
        for tr in tracks:
            all_idx = np.concatenate([d.event_idx for d in tr.detections])
            all_idx = all_idx[all_idx < len(stream.t)]
            tw_all  = stream.t[all_idx].astype(float)

            t_span_ms = (tw_all.max() - tw_all.min()) / 1000.0

            # Spatial x/y spans across full track
            xmin = min(d.x_min for d in tr.detections)
            xmax = max(d.x_max for d in tr.detections)
            ymin = min(d.y_min for d in tr.detections)
            ymax = max(d.y_max for d in tr.detections)
            xy_vol = max((xmax-xmin) * (ymax-ymin), 1.0)

            # 3D event density: events per (px² × ms)
            vol_3d = xy_vol * max(t_span_ms, 0.01)
            ev_density_3d = len(all_idx) / vol_3d

            extras.append([t_span_ms, ev_density_3d])

        extras_arr = np.array(extras, dtype=np.float32)
        return np.hstack([feats_2d, extras_arr])
