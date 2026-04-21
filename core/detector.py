"""
Module 4: detector.py
=====================
Windowed DBSCAN object detector + nearest-neighbour tracker.

Pipeline inside this module:
  1. Divide event stream into short time windows (default 5ms)
  2. Run DBSCAN on each window in (x, y) space → candidate clusters
  3. Link clusters across windows into object trajectories (tracks)
  4. Extract a 15-dimensional feature vector per track

The feature vector is what Module 5 (classifier) uses to predict class.

Key design choices
------------------
• 2D DBSCAN (x, y only) per window — avoids the time-scaling problem
• Greedy nearest-neighbour tracker — simple but robust for ≤20 objects
• Feature extraction captures shape, motion, density, and polarity stats
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Optional
from sklearn.cluster import DBSCAN
from event_reader import EventStream

# ADBScan — drop-in replacement for DBSCAN
try:
    from adbscan import ADBScan, DEFAULT_COEFF_2, DEFAULT_COEFF_1, DEFAULT_BASE
    ADBSCAN_AVAILABLE = True
except ImportError:
    ADBSCAN_AVAILABLE = False
    DEFAULT_COEFF_2 = DEFAULT_COEFF_1 = DEFAULT_BASE = 0.0

try:
    from fast_dbscan import FastDBSCAN
    FAST_DBSCAN_AVAILABLE = True
except ImportError:
    FAST_DBSCAN_AVAILABLE = False


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Detection:
    """Single cluster found by DBSCAN in one time window."""
    window_id   : int
    t_center    : float          # centre timestamp of the window (µs)
    cx          : float          # centroid x (pixels)
    cy          : float          # centroid y (pixels)
    x_min       : float
    x_max       : float
    y_min       : float
    y_max       : float
    n_events    : int
    pol_ratio   : float          # fraction of +1 events
    event_idx   : np.ndarray     # indices into the original stream


@dataclass
class Track:
    """Object trajectory built by linking Detections across windows."""
    track_id    : int
    detections  : List[Detection] = field(default_factory=list)

    @property
    def last(self) -> Detection:
        return self.detections[-1]

    @property
    def length(self) -> int:
        return len(self.detections)

    @property
    def gt_class(self) -> int:
        """Ground-truth class label (if available), else -1."""
        return getattr(self, '_gt_class', -1)

    @gt_class.setter
    def gt_class(self, val):
        self._gt_class = int(val)


# ─────────────────────────────────────────────────────────────────────────────
class EventDetector:
    """
    Windowed DBSCAN object detector + nearest-neighbour tracker.

    Parameters
    ----------
    window_ms      : float   time window width in ms
    eps_px         : float   DBSCAN spatial search radius in pixels
    min_samples    : int     minimum events to form a cluster
    min_track_len  : int     discard tracks shorter than this (noise)
    max_gap_windows: int     max consecutive missed windows before track ends
    track_radius_px: float   max centroid displacement to link detections
    """

    CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist', 'Ball', 'Drone', 'Truck']

    def __init__(
        self,
        window_ms         = 5.0,
        eps_px            = 10.0,
        min_samples       = 5,
        min_track_len     = 4,
        max_gap_windows   = 3,
        track_radius_px   = 40.0,
        use_adbscan       = False,
        use_fast_dbscan   = False,
        adbscan_params    = None,
        verbose           = True,
    ):
        self.window_us        = window_ms * 1000.0
        self.eps_px           = eps_px
        self.min_samples      = min_samples
        self.min_track_len    = min_track_len
        self.max_gap_windows  = max_gap_windows
        self.track_radius_px  = track_radius_px
        self.use_adbscan      = use_adbscan and ADBSCAN_AVAILABLE
        self.verbose          = verbose
        self.use_fast_dbscan  = use_fast_dbscan and FAST_DBSCAN_AVAILABLE \
                                and not use_adbscan

        if adbscan_params is None:
            self.adbscan_params = (0.0001, -0.02, float(eps_px))
        else:
            self.adbscan_params = adbscan_params

        if use_adbscan and not ADBSCAN_AVAILABLE:
            print("  [WARNING] adbscan.py not found — falling back to DBSCAN")
        if use_fast_dbscan and not FAST_DBSCAN_AVAILABLE:
            print("  [WARNING] fast_dbscan.py not found — using sklearn DBSCAN")

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, stream: EventStream, gt_labels: Optional[np.ndarray] = None):
        """
        Run detection on a full event stream.

        Parameters
        ----------
        stream    : EventStream   (filtered events)
        gt_labels : np.ndarray    optional ground-truth class per event (-1=noise)

        Returns
        -------
        tracks   : List[Track]         object trajectories
        features : np.ndarray (N, 15)  feature vector per track
        """
        algo = ("ADBScan" if self.use_adbscan
                else "FastDBSCAN" if self.use_fast_dbscan
                else "DBSCAN")
        print(f"\n[EventDetector] algo={algo} | "
              f"window={self.window_us/1000:.1f}ms | "
              f"eps={self.eps_px}px | "
              f"min_samples={self.min_samples}")
        if self.verbose: print(f"  Input: {len(stream):,} events")
        start = time.time()

        # ── Step 1: Windowed DBSCAN ──────────────────────────────────────────
        raw_detections = self._windowed_dbscan(stream, gt_labels)
        if self.verbose: print(f"  Raw detections across all windows: {len(raw_detections)}")

        # ── Step 2: Track linking ────────────────────────────────────────────
        all_tracks = self._link_tracks(raw_detections)
        valid_tracks = [tr for tr in all_tracks
                        if tr.length >= self.min_track_len]
        if self.verbose: print(f"  Tracks (≥{self.min_track_len} windows): {len(valid_tracks)}")

        # ── Step 3: Assign ground-truth labels ──────────────────────────────
        if gt_labels is not None:
            self._assign_gt_labels(valid_tracks, stream, gt_labels)
            cls_counts = {}
            for tr in valid_tracks:
                name = self.CLASS_NAMES[tr.gt_class] \
                       if 0 <= tr.gt_class < len(self.CLASS_NAMES) else '?'
                cls_counts[name] = cls_counts.get(name, 0) + 1
            for name, cnt in sorted(cls_counts.items()):
                if self.verbose: print(f"    {name:12s}: {cnt} tracks")

        # ── Step 4: Feature extraction ───────────────────────────────────────
        features = self._extract_features(valid_tracks)
        if self.verbose: print(f"  Feature matrix: {features.shape}")
        if self.verbose: print(f"  Time: {time.time()-start:.2f}s")

        return valid_tracks, features

    # ── Step 1: Windowed clustering (DBSCAN or ADBScan) ──────────────────────

    def _windowed_dbscan(self, stream, gt_labels):
        """
        Run DBSCAN or ADBScan on each time window independently.

        Optimisations:
          1. ADBScan object created ONCE outside the window loop
          2. KDTree built once per window and passed into fit_predict()
          3. Window slices pre-computed before clustering loop
        """
        from scipy.spatial import KDTree

        t0        = stream.t[0]
        t_end     = stream.t[-1]
        n_windows = int((t_end - t0) / self.window_us) + 1
        sensor_origin = [stream.W / 2.0, stream.H / 2.0]

        # ── Create clustering objects ONCE outside loop ───────────────────
        if self.use_adbscan:
            adb = ADBScan(mode='pixel',
                          adaptive_params=self.adbscan_params,
                          verbose=False)
        elif self.use_fast_dbscan:
            fdb = FastDBSCAN(eps=self.eps_px, min_samples=self.min_samples)

        detections = []
        win_id = 0

        for w in range(n_windows):
            t_start = t0 + w * self.window_us
            t_stop  = t_start + self.window_us
            mask    = (stream.t >= t_start) & (stream.t < t_stop)

            if mask.sum() < self.min_samples:
                win_id += 1
                continue

            idx_w  = np.where(mask)[0]
            xw     = stream.x[idx_w].astype(np.float64)
            yw     = stream.y[idx_w].astype(np.float64)
            pw     = stream.p[idx_w]
            coords = np.column_stack([xw, yw])

            # ── Choose clustering algorithm ───────────────────────────────
            if self.use_adbscan:
                tree   = KDTree(coords)
                labels = adb.fit_predict(coords, tree=tree)
            elif self.use_fast_dbscan:
                tree   = KDTree(coords)
                labels = fdb.fit_predict(coords, tree=tree)
            else:
                labels = DBSCAN(
                    eps         = self.eps_px,
                    min_samples = self.min_samples,
                    algorithm   = 'ball_tree',
                    n_jobs      = -1,
                ).fit(coords).labels_

            # ── Extract detections ────────────────────────────────────────
            for cid in set(labels):
                if cid == -1:
                    continue
                cm = labels == cid
                if cm.sum() < self.min_samples:
                    continue
                cxv = xw[cm]; cyv = yw[cm]; cpv = pw[cm]
                detections.append(Detection(
                    window_id = win_id,
                    t_center  = (t_start + t_stop) / 2.0,
                    cx        = cxv.mean(),
                    cy        = cyv.mean(),
                    x_min     = cxv.min(),
                    x_max     = cxv.max(),
                    y_min     = cyv.min(),
                    y_max     = cyv.max(),
                    n_events  = int(cm.sum()),
                    pol_ratio = float((cpv > 0).mean()),
                    event_idx = idx_w[cm],
                ))
            win_id += 1

        return detections

    # ── Step 2: Track linking ─────────────────────────────────────────────────

    def _link_tracks(self, detections: List[Detection]) -> List[Track]:
        """
        Greedy nearest-neighbour tracker.

        For each detection (processed in time order):
          - Find the active track whose last detection is closest in (cx, cy)
            and within track_radius_px pixels.
          - If found: append detection to that track.
          - If not found: start a new track.

        A track is 'active' if its last detection was within
        max_gap_windows windows ago.
        """
        tracks: List[Track] = []
        active: List[int]   = []   # indices into tracks[]

        sorted_dets = sorted(detections, key=lambda d: d.window_id)

        for det in sorted_dets:
            best_tid  = None
            best_dist = self.track_radius_px

            for tid in active:
                last = tracks[tid].last
                # Skip stale tracks
                if det.window_id - last.window_id > self.max_gap_windows:
                    continue
                dist = np.sqrt((det.cx - last.cx)**2 +
                               (det.cy - last.cy)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_tid  = tid

            if best_tid is not None:
                tracks[best_tid].detections.append(det)
            else:
                new_id = len(tracks)
                tracks.append(Track(track_id=new_id, detections=[det]))
                active.append(new_id)

        return tracks

    # ── Step 3: Ground-truth label assignment ─────────────────────────────────

    def _assign_gt_labels(self, tracks, stream, gt_labels):
        """Majority vote of event labels in each track → track GT class."""
        for tr in tracks:
            all_gt = []
            for det in tr.detections:
                lbl = gt_labels[det.event_idx]
                all_gt.extend(lbl[lbl >= 0].tolist())
            if all_gt:
                counts = np.bincount(all_gt, minlength=10)
                tr.gt_class = int(counts.argmax())
            else:
                tr.gt_class = -1

    # ── Step 4: Feature extraction ────────────────────────────────────────────

    def _extract_features(self, tracks: List[Track]) -> np.ndarray:
        """
        Extract a 15-dimensional feature vector per track.

        Features
        --------
         0  obj_width      median per-window bounding box width   (pixels)
         1  obj_height     median per-window bounding box height  (pixels)
         2  obj_aspect     width / height
         3  total_events   total events across all windows
         4  duration_ms    track lifetime in ms
         5  speed_px_ms    total displacement / lifetime
         6  speed_x        |horizontal velocity| px/ms
         7  speed_y        |vertical velocity|   px/ms
         8  direction      atan2(dy, dx) in radians
         9  pol_ratio      mean fraction of +1 events
        10  density        events / (width × height)
        11  cx_norm        normalised centroid x (0=left, 1=right)
        12  cy_norm        normalised centroid y (0=top,  1=bottom)
        13  straightness   1 / trajectory residual (high = straight)
        14  osc_amplitude  vertical oscillation amplitude (pedestrian bounce)
        """
        if not tracks:
            return np.zeros((0, 15), dtype=np.float32)

        # Gather representative bbox per track (average over detections)
        all_bbox = []
        for tr in tracks:
            xs = [d.x_min for d in tr.detections]
            xe = [d.x_max for d in tr.detections]
            ys = [d.y_min for d in tr.detections]
            ye = [d.y_max for d in tr.detections]
            all_bbox.append((
                np.median([b-a for a,b in zip(xs,xe)]),   # width
                np.median([b-a for a,b in zip(ys,ye)]),   # height
                float(np.mean([d.cx for d in tr.detections])),
                float(np.mean([d.cy for d in tr.detections])),
            ))

        # Sensor dims (from first track's event counts — approximate)
        # Use 346×260 as default if not otherwise known
        W = 346; H = 260

        feats = []
        for tr, (w, h, cx_mean, cy_mean) in zip(tracks, all_bbox):
            cxs  = np.array([d.cx        for d in tr.detections])
            cys  = np.array([d.cy        for d in tr.detections])
            tcs  = np.array([d.t_center  for d in tr.detections])
            nev  = np.array([d.n_events  for d in tr.detections])
            prs  = np.array([d.pol_ratio for d in tr.detections])

            duration_ms = (tcs[-1] - tcs[0]) / 1000.0 if len(tcs) > 1 else 1.0

            # Velocity from first-half → second-half centroid
            if len(cxs) > 2:
                half = len(cxs) // 2
                dx = cxs[half:].mean() - cxs[:half].mean()
                dy = cys[half:].mean() - cys[:half].mean()
            else:
                dx = cxs[-1] - cxs[0]
                dy = cys[-1] - cys[0]

            spd   = np.sqrt(dx**2 + dy**2) / (duration_ms + 1e-6)
            spd_x = abs(dx) / (duration_ms + 1e-6)
            spd_y = abs(dy) / (duration_ms + 1e-6)
            direc = np.arctan2(dy, dx)

            # Trajectory straightness
            if len(cxs) > 2:
                coef = np.polyfit(tcs, cxs, 1)
                resid = np.std(cxs - np.polyval(coef, tcs))
                straight = 1.0 / (resid + 1e-6)
            else:
                straight = 1.0

            # Vertical oscillation (FFT amplitude — picks up pedestrian bounce)
            if len(cys) > 4:
                osc_amp = np.abs(np.fft.fft(cys - cys.mean())[1:len(cys)//2]).max()
            else:
                osc_amp = 0.0

            feat = np.array([
                float(w),                           # 0  obj width
                float(h),                           # 1  obj height
                float(w) / (float(h) + 1e-6),       # 2  aspect ratio
                float(nev.sum()),                   # 3  total events
                float(duration_ms),                 # 4  duration ms
                float(spd),                         # 5  speed (px/ms)
                float(spd_x),                       # 6  horizontal speed
                float(spd_y),                       # 7  vertical speed
                float(direc),                       # 8  direction (rad)
                float(prs.mean()),                  # 9  polarity ratio
                float(nev.sum()) / (w * h + 1),    # 10 event density
                float(cx_mean / W),                 # 11 cx normalised
                float(cy_mean / H),                 # 12 cy normalised
                float(straight),                    # 13 straightness
                float(osc_amp),                     # 14 oscillation amp
            ], dtype=np.float32)

            feats.append(feat)

        return np.array(feats, dtype=np.float32)

    # ── Summary bounding boxes ────────────────────────────────────────────────

    @staticmethod
    def track_bbox(track: Track):
        """
        Return a representative bounding box for a track.
        Uses the median window to avoid outlier positions.

        Returns
        -------
        (x_min, y_min, x_max, y_max, cx, cy)
        """
        mid = track.detections[len(track.detections) // 2]
        return (mid.x_min, mid.y_min, mid.x_max, mid.y_max,
                mid.cx, mid.cy)


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from event_reader import EventReader
    import os

    os.makedirs('data', exist_ok=True)

    npz_path = 'data/filtered_events.npz'
    lbl_path = 'data/synthetic_labels.npy'

    if not os.path.exists(npz_path):
        print("No filtered stream found — generating synthetic data...")
        stream, labels = EventReader.generate_synthetic()
        stream.save_npz('data/event_stream.npz')
        np.save('data/synthetic_labels.npy', labels)

        from noise_filter import NoiseFilter
        nf = NoiseFilter(mode='multi_scale')
        stream, keep = nf.filter(stream)
        stream.save_npz(npz_path)
        labels = labels[keep]
        np.save('data/filtered_labels.npy', labels)
    else:
        from event_reader import EventReader
        reader = EventReader(npz_path)
        stream = reader.load()
        labels = np.load(lbl_path) if os.path.exists(lbl_path) else None

    det = EventDetector(window_ms=5, eps_px=10, min_samples=5)
    tracks, features = det.detect(stream, labels)

    np.save('data/track_features.npy', features)
    gt_classes = np.array([tr.gt_class for tr in tracks], dtype=np.int8)
    np.save('data/track_gt_classes.npy', gt_classes)
    print(f"\nFeatures saved → data/track_features.npy  shape={features.shape}")
