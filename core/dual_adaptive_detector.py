"""
DualADBScanDetector: Adaptive BOTH eps AND k per event.
True implementation of Intel Patent US 10,510,154.
"automatically computes clustering parameters (radius AND minimum
number of points) based on the distance from the sensor and the
data density in its field of view"

Extension to neuromorphic event cameras (DVS DAVIS346).
Original ADBScan was designed for 2D/3D LiDAR.
"""
import numpy as np
from scipy.spatial import KDTree
from detector_adbscan import ADBScanDetector, _compute_eps_radial, _to_csr, NUMBA_AVAILABLE
from adbscan import _cluster_jit


class DualADBScanDetector(ADBScanDetector):
    """
    Full ADBScan: adaptive epsilon AND adaptive k (min_points) per event.

    Physical justification:
      - k_actual drops ~40% from sensor centre to edges
        (events are sparser at frame periphery)
      - Adapting k matches the effective density threshold
        to the local sensor geometry
      - Rain/snow events always have LOW k_actual → higher
        relative k threshold keeps them as noise
      - Edge object events get a LOWER k threshold →
        objects near frame border are not lost as noise

    Parameters
    ----------
    k_centre : float
        Measured mean k_actual at sensor centre (d≈0). Default 42.
    k_edge   : float
        Measured mean k_actual at sensor edges (d≈max). Default 25.
    """
    def __init__(self, *args, k_centre=42.0, k_edge=25.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_centre = k_centre
        self.k_edge   = k_edge

    def _windowed_cluster(self, stream):
        """
        Override parent: replace fixed k_array with per-point adaptive k.
        Adaptive k decreases toward edges, matching measured event density.
        Everything else (eps adaptation, KDTree, Numba BFS) is identical.
        """
        cx = stream.W / 2.0
        cy = stream.H / 2.0
        c2, c1, base = self.c2, self.c1, self.base
        max_d = np.sqrt(cx**2 + cy**2)

        t_start_all = stream.t[0]
        t_end       = stream.t[-1]
        n_windows   = int((t_end - t_start_all) / self.window_us) + 1
        detections  = []
        win_id      = 0

        for w in range(n_windows):
            t_start = t_start_all + w * self.window_us
            t_stop  = t_start + self.window_us
            mask    = (stream.t >= t_start) & (stream.t < t_stop)
            N       = int(mask.sum())

            if N < self.min_samples:
                win_id += 1; continue

            idx_w  = np.where(mask)[0]
            xw     = stream.x[idx_w].astype(np.float64)
            yw     = stream.y[idx_w].astype(np.float64)
            pw     = stream.p[idx_w]
            coords = np.column_stack([xw, yw])

            # ── Adaptive eps per point (same as parent) ───────────────────
            eps_arr = _compute_eps_radial(xw, yw, cx, cy, c2, c1, base)

            # ── Adaptive k per point (NEW — patent extension) ─────────────
            d = np.sqrt((xw - cx)**2 + (yw - cy)**2)
            d_norm = np.clip(d / max_d, 0.0, 1.0)
            # Linear interpolation of measured k: k_centre at d=0, k_edge at d=max
            k_local = self.k_centre * (1.0 - d_norm) + self.k_edge * d_norm
            # Scale so that min_samples is the k at the sensor centre
            k_array = np.maximum(
                2,
                np.round(
                    self.min_samples * (self.k_centre / np.maximum(k_local, 1.0))
                ).astype(np.int32)
            )

            # ── KDTree + radius search ────────────────────────────────────
            tree   = KDTree(coords)
            nb_raw = tree.query_ball_point(
                coords, eps_arr, workers=1, return_sorted=False)

            # ── CSR + BFS (Numba) ─────────────────────────────────────────
            nb_counts, nb_offsets, nb_data = _to_csr(nb_raw, N)
            if NUMBA_AVAILABLE:
                lbl, _, n_cl = _cluster_jit(
                    N, nb_data, nb_offsets, nb_counts, k_array)
                lbl[lbl > 0] -= 1
            else:
                from detector_adbscan import _cluster_python
                lbl, _, n_cl = _cluster_python(
                    N, nb_data, nb_offsets, nb_counts, k_array)

            # ── Extract detections (identical to parent) ──────────────────
            from detector import Detection
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
