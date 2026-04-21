"""
adbscan.py  —  Adaptive DBSCAN
================================
Rita Chattopadhyay, Intel Corporation — US Patent No. 10,510,154

Two-Parameter Adaptive Computation
------------------------------------
Standard DBSCAN uses FIXED global ε and k.
ADBScan computes BOTH per-point adaptively:

STEP 1 — k (min_points) from quadratic of x-coordinate:
    k(x)  = coeff_2 × x²  +  coeff_1 × x  +  base
    k      = floor(k(x))
    Coefficients from adaptive_params_default.txt:
        coeff_2 = 0.0000404646986111648
        coeff_1 = -0.0161696678274293
        base    = 3.56171908786356

STEP 2 — ε (epsilon) from k via hypersphere volume equation:
    ε  =  0.7 × [ (V × k × Γ(n/2+1)) / (m × √(π^n)) ]^(1/n)
    ε  =  clip(ε, min=0.9)
    where V = np.prod(np.max(X.T) − np.min(X.T))  [global scalar range]
          n = number of dimensions
          m = number of points in current window
          Γ = Gamma function

Domain Notes
------------
The reference equations (compute_cluster_parameters) were designed for
3D LiDAR point clouds in metres.  For 2D event camera pixels the
hypersphere equation produces ε < 0.9px (always hits the clip floor),
so a pixel-adapted mode is also provided.

Two modes:
  mode='lidar'  — exact reference equations (3D metres)
  mode='pixel'  — k from same quadratic; ε = k_cont (pixel units)
"""

import numpy as np
import math
from scipy.spatial import KDTree
from collections import deque
import time
from typing import Tuple

# ── Numba JIT ─────────────────────────────────────────────────────────────────
try:
    from numba import njit, uint8, int32, int8
    NUMBA_AVAILABLE = True

    @njit(cache=True)
    def _cluster_jit(N, nb_data, nb_offsets, nb_counts, k_array):
        """BFS clustering — per-point k threshold — native machine code."""
        labels    = np.zeros(N, dtype=int32)
        core_mask = np.zeros(N, dtype=int8)
        touched   = np.zeros(N, dtype=uint8)
        active    = np.zeros(N, dtype=uint8)
        queue     = np.zeros(N, dtype=int32)
        q_head = 0; q_tail = 0
        no = np.int32(1)

        for i in range(N):
            if touched[i]:
                continue
            cnt_i = nb_counts[i]
            k_i   = k_array[i]
            if cnt_i <= k_i:
                labels[i] = np.int32(-1); core_mask[i] = np.int8(-1)
                touched[i] = np.uint8(1)
                continue

            core_mask[i] = np.int8(1); labels[i] = no
            start_i = nb_offsets[i]; end_i = start_i + cnt_i
            for jj in range(start_i, end_i):
                labels[nb_data[jj]] = no

            q_head = 0; q_tail = 0
            for jj in range(start_i, end_i):
                j = nb_data[jj]
                if active[j] == 0:
                    active[j] = np.uint8(1); queue[q_tail] = j; q_tail += 1

            while q_head < q_tail:
                ind = queue[q_head]; q_head += 1
                touched[ind] = np.uint8(1); active[ind] = np.uint8(0)
                cnt_ind = nb_counts[ind]; k_ind = k_array[ind]
                s2 = nb_offsets[ind]; e2 = s2 + cnt_ind
                if cnt_ind < 2:
                    continue
                core_mask[ind] = np.int8(1) if cnt_ind >= k_ind+1 else np.int8(0)
                for jj in range(s2, e2):
                    j = nb_data[jj]; labels[j] = no
                    if touched[j] == 0:
                        touched[j] = np.uint8(1)
                        if active[j] == 0:
                            active[j] = np.uint8(1)
                            queue[q_tail] = j; q_tail += 1
            no = np.int32(no + 1)

        for i in range(N):
            if labels[i] == 0:
                labels[i] = np.int32(-1); core_mask[i] = np.int8(-1)

        return labels, core_mask, int(no - 1)

except ImportError:
    NUMBA_AVAILABLE = False


# Default coefficients from adaptive_params_default.txt
DEFAULT_COEFF_2 = 0.0000404646986111648
DEFAULT_COEFF_1 = -0.0161696678274293
DEFAULT_BASE    = 3.56171908786356


class ADBScan:
    """
    Adaptive DBSCAN clustering.

    Parameters
    ----------
    mode : 'lidar' or 'pixel'
        'lidar'  — exact reference equations for 3D LiDAR data in metres.
        'pixel'  — adapted for 2D event camera pixel coordinates.
                   k computed by quadratic; ε = k_cont (same quadratic
                   before floor) clipped to min=0.9px.  Both k and ε vary
                   adaptively with the x-pixel position of each point.

    adaptive_params : (coeff_2, coeff_1, base) or None
        Quadratic coefficients. None uses defaults for each mode.
        For mode='pixel', base ≈ desired eps at x=0 (e.g. base=10.0).

    verbose : bool
    """

    def __init__(
        self,
        mode            : str   = 'pixel',
        adaptive_params : Tuple = None,
        verbose         : bool  = False,
    ):
        self.mode    = mode
        self.verbose = verbose

        if adaptive_params is None:
            if mode == 'lidar':
                self.coeff_2, self.coeff_1, self.base = (
                    DEFAULT_COEFF_2, DEFAULT_COEFF_1, DEFAULT_BASE)
            else:  # pixel
                self.coeff_2 = 0.0001
                self.coeff_1 = -0.02
                self.base    = 10.0
        else:
            self.coeff_2, self.coeff_1, self.base = adaptive_params

        self.labels_     = None
        self.core_mask_  = None
        self.eps_array_  = None
        self.k_array_    = None
        self.n_clusters_ = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_predict(self, X: np.ndarray, tree: KDTree = None) -> np.ndarray:
        """
        Run ADBScan on point array X. Returns per-point labels (-1 = noise).

        Parameters
        ----------
        X    : np.ndarray (N, D)   coordinates
        tree : KDTree or None      pre-built tree (pass in for speed)
        """
        X = np.asarray(X, dtype=np.float64)
        N = len(X)
        if N == 0:
            return np.array([], dtype=np.int32)

        t0 = time.time()

        # Step 1 + 2: compute per-point k and eps
        k_array, eps_array = self._compute_k_and_eps(X)
        self.k_array_   = k_array
        self.eps_array_ = eps_array

        if self.verbose:
            print(f"  [ADBScan] N={N} mode={self.mode} "
                  f"k=[{k_array.min()},{k_array.max()}] "
                  f"eps=[{eps_array.min():.2f},{eps_array.max():.2f}]")

        # Build KDTree once
        if tree is None:
            tree = KDTree(X)

        # Batch radius search — each point uses its own eps
        neighbors_raw = tree.query_ball_point(X, eps_array, workers=-1)

        # CSR format for Numba
        nb_counts  = np.array([len(nb) for nb in neighbors_raw], dtype=np.int32)
        nb_offsets = np.zeros(N + 1, dtype=np.int32)
        np.cumsum(nb_counts, out=nb_offsets[1:])
        nb_data = np.empty(int(nb_offsets[N]), dtype=np.int32)
        for i, nb in enumerate(neighbors_raw):
            s = nb_offsets[i]; e = nb_offsets[i+1]
            if e > s:
                nb_data[s:e] = nb

        # BFS clustering with per-point k
        if NUMBA_AVAILABLE:
            labels, core_mask, n_clusters = _cluster_jit(
                N, nb_data, nb_offsets, nb_counts, k_array)
            labels[labels > 0] -= 1      # remap 1-based → 0-based
            self.n_clusters_ = n_clusters
            self.core_mask_  = core_mask
        else:
            labels = self._cluster_python(
                N, nb_data, nb_offsets, nb_counts, k_array)

        if self.verbose:
            print(f"  [ADBScan] Total: {time.time()-t0:.3f}s  "
                  f"Clusters: {self.n_clusters_}  "
                  f"Noise: {(labels==-1).sum()}")

        self.labels_ = labels
        return labels

    # ── k and eps computation ─────────────────────────────────────────────────

    def _compute_k_and_eps(self, X: np.ndarray):
        """
        Compute per-point k and eps.

        Both modes share the same quadratic for k:
            k(x) = c2*x^2 + c1*x + base,  k = floor(k(x))

        where x = X[:,0]  (x-coordinate, first column of point array).

        mode='lidar':  eps from hypersphere volume equation (exact reference).
            V = np.prod(np.max(X.T) - np.min(X.T))   [global scalar]
            eps = 0.7 * [(V*k*Gamma(n/2+1)) / (m*sqrt(pi^n))]^(1/n)
            eps = clip(eps, min=0.9)
            Produces eps in metres — correct for 3D LiDAR.

        mode='pixel':  eps = k_cont (continuous k before floor).
            eps = clip(c2*x^2 + c1*x + base, min=0.9)
            This gives eps in pixel units varying with x-position.
            Physically: wider search radius for points at larger x (far edge),
            tighter for points at small x (near sensor left edge).
        """
        x_coord = X[:, 0].astype(np.float64)
        k_cont  = self.coeff_2 * x_coord**2 + self.coeff_1 * x_coord + self.base
        k_array = np.maximum(np.floor(k_cont).astype(np.int32), 1)

        if self.mode == 'lidar':
            # Exact reference: compute_cluster_parameters()
            xt        = X.T
            n, m      = xt.shape
            V         = float(np.prod(np.max(xt) - np.min(xt)))  # global scalar
            gamma_val = math.gamma(0.5 * n + 1)
            eps = 0.7 * ((V * k_array * gamma_val) /
                         (m * math.sqrt(math.pi ** n))) ** (1.0 / n)
            eps = np.clip(eps, a_min=0.9, a_max=None)
        else:
            # Pixel mode: eps = continuous k (same quadratic, not floored)
            eps = np.clip(k_cont, a_min=0.9, a_max=None)

        return k_array, eps.astype(np.float64)

    # ── Pure Python BFS fallback (no Numba) ───────────────────────────────────

    def _cluster_python(self, N, nb_data, nb_offsets, nb_counts, k_array):
        labels    = np.zeros(N, dtype=np.int32)
        core_mask = np.zeros(N, dtype=np.int8)
        touched   = np.zeros(N, dtype=np.uint8)
        active    = np.zeros(N, dtype=np.uint8)
        no = 1

        for i in range(N):
            if touched[i]:
                continue
            cnt_i = int(nb_counts[i]); k_i = int(k_array[i])
            if cnt_i <= k_i:
                labels[i] = -1; core_mask[i] = -1; touched[i] = 1
                continue
            core_mask[i] = 1; labels[i] = no
            s = nb_offsets[i]; e = nb_offsets[i+1]
            nb_i = nb_data[s:e]; labels[nb_i] = no
            queue = deque(nb_i.tolist()); active[nb_i] = 1

            while queue:
                ind = queue.popleft(); touched[ind] = 1; active[ind] = 0
                s2 = nb_offsets[ind]; e2 = nb_offsets[ind+1]
                nb_ind = nb_data[s2:e2]
                cnt = int(nb_counts[ind]); k_ind = int(k_array[ind])
                if cnt < 2:
                    continue
                core_mask[ind] = 1 if cnt >= k_ind+1 else 0
                labels[nb_ind] = no
                new_mask = touched[nb_ind] == 0
                if new_mask.any():
                    new_pts = nb_ind[new_mask]; touched[new_pts] = 1
                    na = active[new_pts] == 0
                    if na.any():
                        to_add = new_pts[na]; active[to_add] = 1
                        queue.extend(to_add.tolist())
            no += 1

        unassigned = labels == 0
        labels[unassigned] = -1; core_mask[unassigned] = -1
        labels[labels > 0] -= 1
        self.n_clusters_ = no - 1; self.core_mask_ = core_mask
        return labels

    @property
    def n_clusters(self):
        return self.n_clusters_

    @property
    def eps_stats(self):
        if self.eps_array_ is None:
            return {}
        return {'min': float(self.eps_array_.min()),
                'max': float(self.eps_array_.max()),
                'mean': float(self.eps_array_.mean())}
