"""
Module 2: noise_filter.py
=========================
Background Activity (BA) noise filter with three modes:

  Mode 1 — Standard BA Filter
      For each event: check if any neighbouring pixel fired within
      delta_t microseconds. Keep if yes, discard if no.

  Mode 2 — Multi-Scale BA Filter (recommended)
      Runs BA at multiple spatial radii simultaneously.
      Keeps events that have neighbours at ANY scale.
      Better handles mixed object sizes (car + ball in same scene).

  Mode 3 — Density-Adaptive Filter
      Measures local event density and adjusts the search radius.
      High-density regions (large/fast objects) → small radius.
      Low-density regions (small/slow objects)  → larger radius.

Usage:
    from noise_filter import NoiseFilter
    from event_reader import EventReader

    stream, _ = EventReader.generate_synthetic()
    nf        = NoiseFilter(mode='multi_scale')
    clean     = nf.filter(stream)
"""

import numpy as np
import time
from event_reader import EventStream, DEFAULT_WIDTH, DEFAULT_HEIGHT


class NoiseFilter:
    """
    Background Activity noise filter for event streams.

    Parameters
    ----------
    mode : str
        'standard'       — single radius BA filter (fastest)
        'multi_scale'    — BA at 3 radii (recommended)
        'density_adaptive' — adaptive radius based on local density
    delta_t_us : float
        Time window in microseconds.
        Events with no neighbours within this window are noise.
    neighbor_dist : int
        Pixel search radius for 'standard' mode.
    radii : list[int]
        Search radii for 'multi_scale' mode (default [2, 5, 10]).
    min_neighbors : int
        Minimum number of recent neighbours required to keep event.
        Default=1 (standard BA behaviour).
    """

    MODES = ('standard', 'multi_scale', 'density_adaptive')

    def __init__(
        self,
        mode           = 'multi_scale',
        delta_t_us     = 8_000,
        neighbor_dist  = 2,
        radii          = None,
        min_neighbors  = 1,
        verbose        = True,
    ):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.mode          = mode
        self.delta_t_us    = delta_t_us
        self.neighbor_dist = neighbor_dist
        self.radii         = radii or [2, 5, 10]
        self.min_neighbors = min_neighbors
        self.verbose       = verbose

    # ── Public ────────────────────────────────────────────────────────────────

    def filter(self, stream: EventStream) -> EventStream:
        """
        Apply noise filter to an EventStream.

        Returns
        -------
        EventStream containing only the kept (signal) events.
        Also prints a summary of how many events were removed.
        """
        if self.verbose:
            print(f"\n[NoiseFilter] Mode: {self.mode} | "
                  f"delta_t={self.delta_t_us}µs | "
                  f"radii={self.radii if self.mode=='multi_scale' else self.neighbor_dist}px")
            print(f"  Input: {len(stream):,} events")

        start = time.time()

        if self.mode == 'standard':
            keep = self._ba_filter(stream, self.neighbor_dist)

        elif self.mode == 'multi_scale':
            # Keep event if it passes BA at ANY radius
            keep = np.zeros(len(stream), dtype=bool)
            for r in self.radii:
                keep |= self._ba_filter(stream, r)

        elif self.mode == 'density_adaptive':
            keep = self._density_adaptive_filter(stream)

        elapsed = time.time() - start
        n_kept    = keep.sum()
        n_removed = (~keep).sum()
        pct_kept  = n_kept / len(stream) * 100

        if self.verbose:
            print(f"  Kept   : {n_kept:,}  ({pct_kept:.1f}%)")
            print(f"  Removed: {n_removed:,}  ({100-pct_kept:.1f}%)")
            print(f"  Time   : {elapsed:.2f}s")

        return EventStream(
            stream.t[keep], stream.x[keep],
            stream.y[keep], stream.p[keep],
            stream.W, stream.H
        ), keep

    # ── Filter implementations ────────────────────────────────────────────────

    def _ba_filter(self, stream: EventStream, radius: int) -> np.ndarray:
        """
        Core Background Activity filter at a fixed pixel radius.

        Algorithm
        ---------
        Maintain a 2D array  last_timestamp[H, W]
        where last_timestamp[y, x] = the most recent time pixel (x,y) fired.

        For each event (t_i, x_i, y_i):
          1. Extract the (2*radius+1) × (2*radius+1) neighbourhood
             of last_timestamp around (x_i, y_i).
          2. If any neighbour fired after  t_i - delta_t_us → keep.
          3. Update last_timestamp[y_i, x_i] = t_i  (always, keep or not).

        The array lookup is O(radius²) per event — very fast for radius≤10.

        Returns
        -------
        keep : np.ndarray of bool, shape (N,)
        """
        H, W = stream.H, stream.W
        dt   = self.delta_t_us

        # Initialise all timestamps to -infinity (no pixel has fired yet)
        last_ts = np.full((H, W), -1e18, dtype=np.float64)
        keep    = np.zeros(len(stream), dtype=bool)

        t_arr = stream.t
        x_arr = stream.x.astype(np.int32)
        y_arr = stream.y.astype(np.int32)

        for i in range(len(stream)):
            ti = t_arr[i]
            xi = x_arr[i]
            yi = y_arr[i]

            # Clip neighbourhood to sensor bounds
            x0 = max(0, xi - radius);  x1 = min(W, xi + radius + 1)
            y0 = max(0, yi - radius);  y1 = min(H, yi + radius + 1)

            # Check: any neighbour fired within delta_t?
            if np.any(last_ts[y0:y1, x0:x1] > ti - dt):
                keep[i] = True

            last_ts[yi, xi] = ti   # update regardless

        return keep

    def _density_adaptive_filter(self, stream: EventStream) -> np.ndarray:
        """
        Adaptive BA filter: adjusts search radius based on local density.

        Dense regions (large/fast objects)  → small radius (tight clusters)
        Sparse regions (small objects/noise) → large radius (spread out)

        Uses a coarse 5-pixel density map updated as events arrive.
        """
        H, W  = stream.H, stream.W
        dt    = self.delta_t_us

        last_ts  = np.full((H, W), -1e18, dtype=np.float64)
        density  = np.zeros((H, W), dtype=np.float32)  # rolling activity count
        keep     = np.zeros(len(stream), dtype=bool)
        DECAY    = 0.995   # decay density estimate each event

        t_arr = stream.t
        x_arr = stream.x.astype(np.int32)
        y_arr = stream.y.astype(np.int32)

        for i in range(len(stream)):
            ti = t_arr[i]; xi = x_arr[i]; yi = y_arr[i]

            # Estimate local density in a 9×9 neighbourhood
            xd0 = max(0, xi-4); xd1 = min(W, xi+5)
            yd0 = max(0, yi-4); yd1 = min(H, yi+5)
            local_density = density[yd0:yd1, xd0:xd1].mean()

            # Choose radius based on density
            if local_density > 40:
                r = 2     # dense — tight radius
            elif local_density > 15:
                r = 4     # medium
            else:
                r = 8     # sparse — wide radius

            x0 = max(0, xi-r); x1 = min(W, xi+r+1)
            y0 = max(0, yi-r); y1 = min(H, yi+r+1)

            if np.any(last_ts[y0:y1, x0:x1] > ti - dt):
                keep[i] = True

            last_ts[yi, xi] = ti
            density *= DECAY
            density[yi, xi] += 1.0

        return keep


# ─── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    from event_reader import EventReader
    import argparse, os

    parser = argparse.ArgumentParser(description="Event stream noise filter")
    parser.add_argument('--input',    type=str, default='data/event_stream.npz')
    parser.add_argument('--output',   type=str, default='data/filtered_events.npz')
    parser.add_argument('--mode',     type=str, default='multi_scale',
                        choices=NoiseFilter.MODES)
    parser.add_argument('--delta_t',  type=int, default=8000,
                        help="Time window in µs (default 8000 = 8ms)")
    args = parser.parse_args()

    # Generate data if input doesn't exist
    if not os.path.exists(args.input):
        print(f"Input file not found. Generating synthetic data...")
        stream, labels = EventReader.generate_synthetic()
        stream.save_npz(args.input)
        np.save('data/synthetic_labels.npy', labels)
    else:
        reader = EventReader(args.input)
        stream = reader.load()

    nf = NoiseFilter(mode=args.mode, delta_t_us=args.delta_t)
    clean_stream, keep_mask = nf.filter(stream)
    clean_stream.save_npz(args.output)
    np.save(args.output.replace('.npz', '_keep_mask.npy'), keep_mask)
    print(f"\nFiltered stream saved → {args.output}")
