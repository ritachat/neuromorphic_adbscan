"""
Module 1: event_reader.py
=========================
Universal reader for neuromorphic event camera data.

Supports three common formats:
  1. CSV / TXT  — columns: timestamp_us, x, y, polarity
  2. NumPy .npz — arrays: t, x, y, p  (saved by our pipeline)
  3. Binary .bin — raw packed uint64 events (iniVation / RPG format)

Also includes:
  - Synthetic data generator (used when no real dataset available)
  - Event stream statistics reporter
  - Time-window slicing

Usage:
    reader = EventReader('data/my_recording.csv')
    stream = reader.load()
    window = stream.slice_time(0, 50_000)   # first 50ms
"""

import numpy as np
import os
import struct
import time

# ─── Sensor defaults (override per your hardware) ────────────────────────────
DEFAULT_WIDTH  = 346   # DAVIS346
DEFAULT_HEIGHT = 260

# ─────────────────────────────────────────────────────────────────────────────
class EventStream:
    """
    Container for a stream of events.

    Attributes
    ----------
    t : np.ndarray  dtype=float64   timestamps in microseconds
    x : np.ndarray  dtype=int16     pixel column  (0 → W-1)
    y : np.ndarray  dtype=int16     pixel row     (0 → H-1)
    p : np.ndarray  dtype=int8      polarity  +1 or -1
    W, H : int      sensor dimensions
    """

    def __init__(self, t, x, y, p, W=DEFAULT_WIDTH, H=DEFAULT_HEIGHT):
        order = np.argsort(t)           # always keep events time-sorted
        self.t = np.asarray(t, dtype=np.float64)[order]
        self.x = np.asarray(x, dtype=np.int16)[order]
        self.y = np.asarray(y, dtype=np.int16)[order]
        self.p = np.asarray(p, dtype=np.int8)[order]
        self.W = W
        self.H = H

    # ── Basic properties ──────────────────────────────────────────────────────

    def __len__(self):
        return len(self.t)

    @property
    def duration_us(self):
        """Total stream duration in microseconds."""
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0

    @property
    def duration_ms(self):
        return self.duration_us / 1000.0

    @property
    def event_rate(self):
        """Mean event rate in events/second."""
        if self.duration_us == 0:
            return 0.0
        return len(self.t) / (self.duration_us * 1e-6)

    # ── Slicing ───────────────────────────────────────────────────────────────

    def slice_time(self, t_start_us, t_end_us):
        """Return a new EventStream containing events in [t_start, t_end) µs."""
        mask = (self.t >= t_start_us) & (self.t < t_end_us)
        return EventStream(self.t[mask], self.x[mask],
                           self.y[mask], self.p[mask],
                           self.W, self.H)

    def slice_region(self, x0, y0, x1, y1):
        """Return events within a rectangular pixel region."""
        mask = (self.x >= x0) & (self.x < x1) & \
               (self.y >= y0) & (self.y < y1)
        return EventStream(self.t[mask], self.x[mask],
                           self.y[mask], self.p[mask],
                           self.W, self.H)

    def time_windows(self, window_us):
        """
        Generator: yields successive EventStream windows of width window_us.
        Useful for processing a long recording chunk by chunk.
        """
        if len(self) == 0:
            return
        t0 = self.t[0]
        t_end = self.t[-1]
        current = t0
        while current < t_end:
            yield self.slice_time(current, current + window_us)
            current += window_us

    # ── Saving ────────────────────────────────────────────────────────────────

    def save_npz(self, path):
        np.savez_compressed(path,
                            t=self.t, x=self.x, y=self.y, p=self.p,
                            W=np.array(self.W), H=np.array(self.H))
        print(f"  Saved EventStream → {path}  ({len(self):,} events)")

    def save_csv(self, path):
        header = "timestamp_us,x,y,polarity"
        data   = np.column_stack([self.t, self.x, self.y, self.p])
        np.savetxt(path, data, delimiter=',', header=header,
                   comments='', fmt=['%.1f', '%d', '%d', '%d'])
        print(f"  Saved CSV → {path}  ({len(self):,} events)")

    # ── Statistics ────────────────────────────────────────────────────────────

    def summary(self):
        if len(self) == 0:
            print("  EventStream: EMPTY")
            return
        pos = (self.p > 0).sum()
        neg = (self.p < 0).sum()
        print(f"  EventStream Summary")
        print(f"  ├─ Total events    : {len(self):,}")
        print(f"  ├─ Duration        : {self.duration_ms:.1f} ms")
        print(f"  ├─ Event rate      : {self.event_rate/1000:.1f} K events/sec")
        print(f"  ├─ Positive (+1)   : {pos:,}  ({pos/len(self)*100:.1f}%)")
        print(f"  ├─ Negative (-1)   : {neg:,}  ({neg/len(self)*100:.1f}%)")
        print(f"  ├─ Sensor size     : {self.W} × {self.H}")
        print(f"  ├─ x range         : [{self.x.min()}, {self.x.max()}]")
        print(f"  └─ y range         : [{self.y.min()}, {self.y.max()}]")


# ─────────────────────────────────────────────────────────────────────────────
class EventReader:
    """
    Reads event data from multiple file formats and returns an EventStream.

    Parameters
    ----------
    path : str
        Path to the event file. Extension determines format:
        .csv / .txt  → CSV reader
        .npz         → NumPy archive reader
        .bin         → Binary packed-uint64 reader
        None         → Uses the built-in synthetic generator
    W, H : int
        Sensor resolution (used for binary format where not embedded).
    """

    def __init__(self, path=None, W=DEFAULT_WIDTH, H=DEFAULT_HEIGHT):
        self.path = path
        self.W    = W
        self.H    = H

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self):
        """Load and return an EventStream from self.path."""
        if self.path is None:
            raise ValueError("No path given. Use generate_synthetic() instead.")

        ext = os.path.splitext(self.path)[1].lower()
        print(f"\n[EventReader] Loading: {self.path}")
        start = time.time()

        if ext in ('.csv', '.txt'):
            stream = self._read_csv()
        elif ext == '.npz':
            stream = self._read_npz()
        elif ext == '.bin':
            stream = self._read_bin()
        else:
            raise ValueError(f"Unknown extension '{ext}'. "
                             f"Supported: .csv, .txt, .npz, .bin")

        elapsed = time.time() - start
        print(f"  Loaded {len(stream):,} events in {elapsed:.2f}s")
        stream.summary()
        return stream

    # ── Format readers ────────────────────────────────────────────────────────

    def _read_csv(self):
        """
        CSV format expected:
            timestamp_us, x, y, polarity
        First row may be a header (auto-detected).
        Polarity values: +1/1/True → +1   0/-1/False → -1
        """
        import pandas as pd

        # Try reading with header first
        try:
            df = pd.read_csv(self.path)
            # Normalise column names to lowercase
            df.columns = [c.lower().strip() for c in df.columns]

            # Map common column name variants
            col_map = {
                'timestamp_us': 't', 'timestamp': 't', 'ts': 't', 'time': 't',
                'pol': 'p', 'polarity': 'p',
            }
            df.rename(columns=col_map, inplace=True)

            # Verify required columns
            for col in ['t', 'x', 'y', 'p']:
                if col not in df.columns:
                    raise KeyError(f"Column '{col}' not found. "
                                   f"Available: {list(df.columns)}")

            t = df['t'].values.astype(np.float64)
            x = df['x'].values.astype(np.int16)
            y = df['y'].values.astype(np.int16)
            p = df['p'].values

        except Exception:
            # Fall back to headerless reading (raw 4-column numeric)
            data = np.loadtxt(self.path, delimiter=',', skiprows=0)
            t, x, y, p = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

        # Normalise polarity: anything > 0 → +1, otherwise → -1
        p = np.where(p > 0, 1, -1).astype(np.int8)

        return EventStream(t, x, y, p, self.W, self.H)

    def _read_npz(self):
        """NumPy .npz archive with arrays t, x, y, p and optional W, H."""
        data = np.load(self.path, allow_pickle=True)
        W    = int(data['W']) if 'W' in data else self.W
        H    = int(data['H']) if 'H' in data else self.H
        return EventStream(data['t'], data['x'], data['y'], data['p'], W, H)

    def _read_bin(self):
        """
        Raw binary format (iniVation / RPG convention):
        Each event = 8 bytes packed as uint64:
            bits 63-32 : timestamp in microseconds
            bits 31-16 : x coordinate
            bits 15- 1 : y coordinate
            bit       0 : polarity (1=ON, 0=OFF)

        Falls back to simple 4×int32 packing if uint64 decode
        produces out-of-range coordinates.
        """
        raw = np.fromfile(self.path, dtype=np.uint64)

        t = ((raw >> 32) & 0xFFFFFFFF).astype(np.float64)
        x = ((raw >> 16) & 0xFFFF).astype(np.int16)
        y = ((raw >>  1) & 0x7FFF).astype(np.int16)
        p = (raw & 0x1).astype(np.int8)
        p = np.where(p > 0, 1, -1)

        # Sanity check — if coords are out of range try 4×int32 layout
        if x.max() > 4096 or y.max() > 4096:
            print("  [WARN] uint64 decode gave invalid coords. "
                  "Trying 4×int32 layout...")
            raw32 = np.fromfile(self.path, dtype=np.int32).reshape(-1, 4)
            t = raw32[:, 0].astype(np.float64)
            x = raw32[:, 1].astype(np.int16)
            y = raw32[:, 2].astype(np.int16)
            p = np.where(raw32[:, 3] > 0, 1, -1).astype(np.int8)

        return EventStream(t, x, y, p, self.W, self.H)

    # ── Synthetic data generator ──────────────────────────────────────────────

    @staticmethod
    def generate_synthetic(
        duration_ms   = 200,
        n_objects     = 6,
        W             = DEFAULT_WIDTH,
        H             = DEFAULT_HEIGHT,
        noise_rate    = 0.05,
        seed          = 42,
    ):
        """
        Generate a realistic synthetic event stream with multiple
        moving objects of different classes for testing the pipeline.

        Object classes generated:
          0 Car         — wide, fast, horizontal
          1 Pedestrian  — tall, slow, slight bounce
          2 Cyclist     — medium, medium speed
          3 Ball        — small, fast, parabolic arc
          4 Drone       — small, erratic zigzag
          5 Truck       — very wide, slow

        Parameters
        ----------
        duration_ms : int   total recording length in milliseconds
        n_objects   : int   number of object instances (up to 8 placed)
        noise_rate  : float fraction of events that are background noise
        seed        : int   random seed for reproducibility

        Returns
        -------
        stream  : EventStream
        labels  : np.ndarray  ground-truth class per event (-1 = noise)
        """
        rng = np.random.default_rng(seed)
        DURATION_US = duration_ms * 1000

        CLASS_DEFS = {
          # cls: (width, height, speed_x, speed_y, density, pattern)
          0: (55, 22, 3.5, 0.1, 16, 'straight'),   # Car
          1: (16, 48, 0.7, 0.0, 10, 'bounce'),      # Pedestrian
          2: (26, 34, 1.8, 0.1, 13, 'straight'),    # Cyclist
          3: (12, 12, 3.2, 2.5, 8,  'arc'),         # Ball
          4: (13, 10, 2.5, 2.5, 7,  'zigzag'),      # Drone
          5: (80, 30, 1.5, 0.0, 20, 'straight'),    # Truck
        }

        # Place object instances (cycle through classes if n_objects > 6)
        placements = [
            (0, 10,  90),   # Car 1
            (1, 60,  55),   # Pedestrian 1
            (2, 120, 175),  # Cyclist
            (3, 30,  30),   # Ball
            (4, 200, 100),  # Drone
            (5, 10,  200),  # Truck
            (0, 10,  140),  # Car 2
            (1, 180, 40),   # Pedestrian 2
        ]
        placements = placements[:n_objects]

        all_t, all_x, all_y, all_p, all_lbl = [], [], [], [], []
        N_STEPS = 400
        dt = DURATION_US / N_STEPS

        for inst_id, (cls, sx, sy) in enumerate(placements):
            cfg = CLASS_DEFS[cls % len(CLASS_DEFS)]
            OW, OH, spx, spy, dens, pat = cfg

            for step in range(N_STEPS):
                t_now = step * dt
                frac  = step / N_STEPS

                if pat == 'straight':
                    cx = sx + spx * step * 0.55
                    cy = sy + spy * step * 0.55
                elif pat == 'bounce':
                    cx = sx + spx * step * 0.55
                    cy = sy + 8 * np.sin(step * 0.09)
                elif pat == 'arc':
                    cx = sx + spx * step * 0.45
                    cy = sy + spy * step * 0.45 - 0.004 * step ** 2
                elif pat == 'zigzag':
                    cx = sx + spx * step * 0.3 * np.sign(np.sin(step * 0.06))
                    cy = sy + spy * step * 0.3 * np.sign(np.cos(step * 0.06))

                cx += rng.normal(0, 0.4)
                cy += rng.normal(0, 0.4)
                cx  = float(np.clip(cx, OW // 2 + 1, W - OW // 2 - 1))
                cy  = float(np.clip(cy, OH // 2 + 1, H - OH // 2 - 1))

                n_ev = rng.poisson(dens)
                if n_ev == 0:
                    continue

                # Events fire on the object boundary (edges, like real DVS)
                side = rng.integers(0, 4, n_ev)
                ex = np.zeros(n_ev); ey = np.zeros(n_ev)
                ex[side==0] = cx + rng.uniform(-OW/2, OW/2, (side==0).sum())
                ey[side==0] = cy - OH/2 + rng.normal(0, 1.2, (side==0).sum())
                ex[side==1] = cx + rng.uniform(-OW/2, OW/2, (side==1).sum())
                ey[side==1] = cy + OH/2 + rng.normal(0, 1.2, (side==1).sum())
                ex[side==2] = cx - OW/2 + rng.normal(0, 1.2, (side==2).sum())
                ey[side==2] = cy + rng.uniform(-OH/2, OH/2, (side==2).sum())
                ex[side==3] = cx + OW/2 + rng.normal(0, 1.2, (side==3).sum())
                ey[side==3] = cy + rng.uniform(-OH/2, OH/2, (side==3).sum())

                et = t_now + rng.uniform(0, dt, n_ev)
                ep = np.where(side <= 1, 1, -1).astype(np.int8)
                valid = (ex >= 0) & (ex < W) & (ey >= 0) & (ey < H)

                all_t.append(et[valid])
                all_x.append(ex[valid].astype(np.int16))
                all_y.append(ey[valid].astype(np.int16))
                all_p.append(ep[valid])
                all_lbl.append(np.full(valid.sum(), cls, dtype=np.int8))

        # Background noise
        n_total_so_far = sum(len(a) for a in all_t)
        n_noise = int(n_total_so_far * noise_rate)
        all_t.append(rng.uniform(0, DURATION_US, n_noise))
        all_x.append(rng.integers(0, W, n_noise).astype(np.int16))
        all_y.append(rng.integers(0, H, n_noise).astype(np.int16))
        all_p.append(rng.choice([-1, 1], n_noise).astype(np.int8))
        all_lbl.append(np.full(n_noise, -1, dtype=np.int8))

        t_all = np.concatenate(all_t)
        x_all = np.concatenate(all_x)
        y_all = np.concatenate(all_y)
        p_all = np.concatenate(all_p)
        lbl   = np.concatenate(all_lbl)

        order = np.argsort(t_all)
        stream = EventStream(t_all[order], x_all[order],
                             y_all[order], p_all[order], W, H)
        labels = lbl[order]

        print(f"\n[EventReader] Generated synthetic stream:")
        stream.summary()
        print(f"  Object classes : {n_objects} instances")
        print(f"  Noise events   : {n_noise:,}  ({noise_rate*100:.0f}%)")
        return stream, labels


# ─── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse, sys

    parser = argparse.ArgumentParser(description="Event camera file reader")
    parser.add_argument('--file',   type=str, default=None,
                        help="Path to event file (.csv/.npz/.bin). "
                             "Omit to use synthetic data.")
    parser.add_argument('--width',  type=int, default=DEFAULT_WIDTH)
    parser.add_argument('--height', type=int, default=DEFAULT_HEIGHT)
    parser.add_argument('--save',   type=str, default=None,
                        help="Save loaded/generated stream to this .npz path")
    args = parser.parse_args()

    if args.file:
        reader = EventReader(args.file, args.width, args.height)
        stream = reader.load()
    else:
        print("No file given — generating synthetic data...")
        stream, labels = EventReader.generate_synthetic()
        np.save('data/synthetic_labels.npy', labels)

    if args.save:
        stream.save_npz(args.save)
    else:
        stream.save_npz('data/event_stream.npz')

    print("\nDone.")
