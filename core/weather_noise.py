"""
weather_noise.py
================
Realistic rain and snow noise models for neuromorphic event cameras.

Physics behind weather noise on event cameras
----------------------------------------------

RAIN:
  A raindrop falling through the camera's field of view creates a
  moving bright streak. As the leading edge of the drop enters a
  pixel's view → brightness INCREASES → +1 event.
  As the drop passes and the pixel sees the dark background again
  → brightness DECREASES → -1 event.
  This creates characteristic BURST patterns:
    - Short vertical streaks (drop trajectory)
    - High local event density in a small area
    - +1 events slightly ahead of -1 events (leading/trailing edge)
    - Duration per drop: 1–10ms depending on drop size and speed

SNOW:
  Snowflakes are larger and slower than rain.
  They create DIFFUSE, SLOWER events:
    - Irregular blob shapes (not straight streaks)
    - Lower local density than rain
    - Longer duration per flake: 10–50ms
    - More random polarity (light scattering vs absorption)
    - Appear at all depths — nearby flakes are large blobs,
      distant flakes are single-pixel events

Key difference from random noise:
  Random noise  → single isolated pixel events, no spatial structure
  Rain noise    → short burst clusters with vertical directionality
  Snow noise    → slow diffuse blobs with random motion

This makes weather noise HARDER to remove than random noise —
it has spatial-temporal structure that looks similar to real objects.

Usage:
    from weather_noise import WeatherNoise
    from event_reader  import EventReader

    stream, labels = EventReader.generate_synthetic()
    wn = WeatherNoise(rain_intensity='heavy', snow_intensity='light')
    noisy_stream, weather_labels = wn.add_weather(stream, labels)
"""

import numpy as np
from event_reader import EventStream, DEFAULT_WIDTH, DEFAULT_HEIGHT

# Weather label codes (extend the class label space)
LABEL_RAIN  = 10   # rain drop event
LABEL_SNOW  = 11   # snowflake event
LABEL_CLEAN = -1   # original background noise


class WeatherNoise:
    """
    Adds physically realistic rain and/or snow noise to an event stream.

    Parameters
    ----------
    rain_intensity : str or None
        'light'  — occasional drops, low density
        'medium' — moderate rainfall
        'heavy'  — dense rainfall, many overlapping streaks
        None     — no rain

    snow_intensity : str or None
        'light'  — occasional flakes
        'medium' — moderate snowfall
        'heavy'  — blizzard conditions
        None     — no snow

    seed : int   random seed for reproducibility
    """

    # Rain parameters: (drops_per_second, streak_length_px, drop_speed_px_ms)
    RAIN_PARAMS = {
        'light':  dict(drops_per_sec=200,  streak_len=(3,  8),  speed=(8, 15),  cluster_r=1),
        'medium': dict(drops_per_sec=600,  streak_len=(5,  15), speed=(10, 20), cluster_r=2),
        'heavy':  dict(drops_per_sec=1500, streak_len=(8,  25), speed=(12, 25), cluster_r=2),
    }

    # Snow parameters: (flakes_per_second, flake_radius_px, flake_speed_px_ms)
    SNOW_PARAMS = {
        'light':  dict(flakes_per_sec=50,  radius=(2, 6),  speed=(0.5, 2.0), drift=0.5),
        'medium': dict(flakes_per_sec=150, radius=(3, 10), speed=(0.8, 3.0), drift=1.0),
        'heavy':  dict(flakes_per_sec=400, radius=(4, 14), speed=(1.0, 4.0), drift=2.0),
    }

    def __init__(self, rain_intensity=None, snow_intensity=None, seed=123, verbose=False):
        self.rain_intensity = rain_intensity
        self.verbose = verbose  # controlled externally
        self.snow_intensity = snow_intensity
        self.rng = np.random.default_rng(seed)

    # ── Public API ────────────────────────────────────────────────────────────

    def add_weather(self, stream: EventStream,
                    labels: np.ndarray = None):
        """
        Add weather noise events to an existing event stream.

        Parameters
        ----------
        stream : EventStream   clean event stream
        labels : np.ndarray    ground-truth labels per event (optional)

        Returns
        -------
        noisy_stream   : EventStream   original + weather events merged
        weather_labels : np.ndarray    labels for ALL events
                         (original labels preserved, weather events
                          labelled LABEL_RAIN=10 or LABEL_SNOW=11)
        stats          : dict          count of weather events added
        """
        all_t = [stream.t]
        all_x = [stream.x.astype(np.float32)]
        all_y = [stream.y.astype(np.float32)]
        all_p = [stream.p.astype(np.int8)]
        all_lbl = [labels if labels is not None
                   else np.full(len(stream), LABEL_CLEAN, dtype=np.int8)]

        stats = {}

        if self.rain_intensity is not None:
            rt, rx, ry, rp = self._generate_rain(
                stream.duration_us, stream.t[0],
                stream.W, stream.H,
                self.RAIN_PARAMS[self.rain_intensity]
            )
            all_t.append(rt); all_x.append(rx)
            all_y.append(ry); all_p.append(rp)
            all_lbl.append(np.full(len(rt), LABEL_RAIN, dtype=np.int8))
            stats['rain_events'] = len(rt)
            if self.verbose: print(f"  Rain ({self.rain_intensity}): {len(rt):,} events added")

        if self.snow_intensity is not None:
            st, sx, sy, sp = self._generate_snow(
                stream.duration_us, stream.t[0],
                stream.W, stream.H,
                self.SNOW_PARAMS[self.snow_intensity]
            )
            all_t.append(st); all_x.append(sx)
            all_y.append(sy); all_p.append(sp)
            all_lbl.append(np.full(len(st), LABEL_SNOW, dtype=np.int8))
            stats['snow_events'] = len(st)
            if self.verbose: print(f"  Snow ({self.snow_intensity}): {len(st):,} events added")

        # Merge and sort
        t_all   = np.concatenate(all_t)
        x_all   = np.concatenate(all_x).astype(np.int16)
        y_all   = np.concatenate(all_y).astype(np.int16)
        p_all   = np.concatenate(all_p).astype(np.int8)
        lbl_all = np.concatenate(all_lbl).astype(np.int8)

        order   = np.argsort(t_all)
        noisy   = EventStream(t_all[order], x_all[order],
                              y_all[order], p_all[order],
                              stream.W, stream.H)
        weather_labels = lbl_all[order]

        n_weather = len(t_all) - len(stream)
        pct = n_weather / len(noisy) * 100
        stats['total_weather'] = n_weather
        stats['weather_pct']   = pct
        if self.verbose: print(f"  Total weather events: {n_weather:,}  ({pct:.1f}% of stream)")
        return noisy, weather_labels, stats

    # ── Rain generator ────────────────────────────────────────────────────────

    def _generate_rain(self, duration_us, t_offset, W, H, params):
        """
        Simulate raindrops falling through the sensor field of view.

        Each raindrop:
          1. Starts at a random (x, y_top) position
          2. Falls vertically (with slight horizontal drift)
          3. Creates a streak of events along its path
          4. Leading edge (+1 events) slightly ahead of trailing (-1 events)
          5. Burst of events concentrated at entry and exit
        """
        drops_per_sec = params['drops_per_sec']
        streak_range  = params['streak_len']
        speed_range   = params['speed']
        cluster_r     = params['cluster_r']

        duration_ms   = duration_us / 1000.0
        n_drops       = int(drops_per_sec * duration_ms / 1000)

        t_all, x_all, y_all, p_all = [], [], [], []

        for _ in range(n_drops):
            # Random drop properties
            x_start    = self.rng.integers(0, W)
            y_start    = self.rng.integers(0, max(1, H // 3))  # drops start high
            streak_len = self.rng.integers(*streak_range)
            speed      = self.rng.uniform(*speed_range)         # px/ms
            t_start    = t_offset + self.rng.uniform(0, duration_us)
            drift_x    = self.rng.uniform(-0.3, 0.3)            # slight sideways

            # Duration of this drop passing through the streak
            drop_dur_us = (streak_len / speed) * 1000

            # Number of events this drop generates
            n_events = self.rng.poisson(streak_len * 3)
            if n_events == 0:
                continue

            # Distribute events along the streak
            progress = self.rng.uniform(0, 1, n_events)   # 0=top, 1=bottom
            t_ev = t_start + progress * drop_dur_us

            # x position: start + drift
            x_ev = x_start + drift_x * progress * streak_len
            x_ev += self.rng.normal(0, cluster_r, n_events)

            # y position: falls downward
            y_ev = y_start + progress * streak_len
            y_ev += self.rng.normal(0, cluster_r * 0.5, n_events)

            # Polarity: leading edge +1, trailing edge -1
            # Events at top of streak are +1 (pixel brightens as drop arrives)
            # Events at bottom are -1 (pixel darkens as drop leaves)
            p_ev = np.where(progress < 0.5, 1, -1).astype(np.int8)

            # Clip to sensor
            valid = ((x_ev >= 0) & (x_ev < W) &
                     (y_ev >= 0) & (y_ev < H) &
                     (t_ev >= t_offset) & (t_ev < t_offset + duration_us))

            t_all.append(t_ev[valid])
            x_all.append(x_ev[valid].astype(np.int16))
            y_all.append(y_ev[valid].astype(np.int16))
            p_all.append(p_ev[valid])

        if not t_all:
            return (np.array([]), np.array([]),
                    np.array([]), np.array([]))

        return (np.concatenate(t_all),
                np.concatenate(x_all),
                np.concatenate(y_all),
                np.concatenate(p_all))

    # ── Snow generator ────────────────────────────────────────────────────────

    def _generate_snow(self, duration_us, t_offset, W, H, params):
        """
        Simulate snowflakes drifting through the sensor field of view.

        Each snowflake:
          1. Appears at a random position (larger flakes = closer to sensor)
          2. Drifts slowly downward with random horizontal drift
          3. Creates a diffuse BLOB of events (not a sharp streak)
          4. Events fire on the flake's irregular boundary
          5. Polarity is more random (light scattering, not clean edges)
          6. Longer duration than rain (slower movement)
        """
        flakes_per_sec = params['flakes_per_sec']
        radius_range   = params['radius']
        speed_range    = params['speed']
        drift          = params['drift']

        duration_ms    = duration_us / 1000.0
        n_flakes       = int(flakes_per_sec * duration_ms / 1000)

        t_all, x_all, y_all, p_all = [], [], [], []

        for _ in range(n_flakes):
            # Flake properties
            x_start   = self.rng.integers(0, W)
            y_start   = self.rng.integers(0, H)
            radius    = self.rng.integers(*radius_range)
            speed     = self.rng.uniform(*speed_range)
            drift_x   = self.rng.uniform(-drift, drift)
            t_start   = t_offset + self.rng.uniform(0, duration_us)

            # Flake lifetime: time to travel its own diameter
            flake_dur_us = (radius * 2 / speed) * 1000

            # Number of events (blobs are diffuse)
            n_events = self.rng.poisson(radius ** 2 * 2)
            if n_events == 0:
                continue

            # Spread events in time and space (blob shape)
            progress = self.rng.uniform(0, 1, n_events)
            t_ev     = t_start + progress * flake_dur_us

            # Spatial: random distribution within flake radius (blob)
            angles  = self.rng.uniform(0, 2 * np.pi, n_events)
            radii_r = self.rng.uniform(0, radius, n_events)
            x_ev    = x_start + drift_x * progress * radius * 2 \
                      + radii_r * np.cos(angles)
            y_ev    = y_start + progress * radius * 2 \
                      + radii_r * np.sin(angles) * 0.5   # flatter vertically

            # Snow polarity: more random due to light scattering
            # (unlike rain which has clear leading/trailing edges)
            p_ev = self.rng.choice([-1, 1], n_events,
                                   p=[0.45, 0.55]).astype(np.int8)

            # Clip to sensor
            valid = ((x_ev >= 0) & (x_ev < W) &
                     (y_ev >= 0) & (y_ev < H) &
                     (t_ev >= t_offset) & (t_ev < t_offset + duration_us))

            t_all.append(t_ev[valid])
            x_all.append(x_ev[valid].astype(np.int16))
            y_all.append(y_ev[valid].astype(np.int16))
            p_all.append(p_ev[valid])

        if not t_all:
            return (np.array([]), np.array([]),
                    np.array([]), np.array([]))

        return (np.concatenate(t_all),
                np.concatenate(x_all),
                np.concatenate(y_all),
                np.concatenate(p_all))


# ─────────────────────────────────────────────────────────────────────────────
def visualise_weather(stream_clean, stream_noisy,
                      labels_clean, labels_noisy,
                      title='Weather Noise Comparison'):
    """
    Generate a 3-panel comparison plot:
      Left  : Clean stream (objects only)
      Middle: Weather noise only
      Right : Combined (objects + weather)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import os

    os.makedirs('plots', exist_ok=True)

    BG    = '#0a0e1a'
    PANEL = '#111827'
    TEXT  = '#e2e8f0'
    MUTED = '#9ca3af'

    CLASS_COLOURS = {
        0:  '#ef4444',   # Car
        1:  '#22c55e',   # Pedestrian
        2:  '#f59e0b',   # Cyclist
        3:  '#a855f7',   # Ball
        4:  '#06b6d4',   # Drone
        5:  '#f97316',   # Truck
       -1:  '#374151',   # Random noise
        10: '#60a5fa',   # Rain  — light blue
        11: '#e2e8f0',   # Snow  — white/silver
    }
    CLASS_NAMES = {
        0:'Car', 1:'Pedestrian', 2:'Cyclist',
        3:'Ball', 4:'Drone', 5:'Truck',
       -1:'Noise', 10:'Rain', 11:'Snow'
    }

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    fig.patch.set_facecolor(BG)

    rng = np.random.default_rng(0)

    def scatter_stream(ax, t, x, y, p, lbl, title_str, max_pts=8000):
        ax.set_facecolor(PANEL)
        ax.set_title(title_str, color=TEXT, fontsize=12,
                     fontweight='bold', pad=8)
        ax.set_xlim(0, stream_noisy.W)
        ax.set_ylim(stream_noisy.H, 0)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel('x pixel', color=MUTED)
        ax.set_ylabel('y pixel', color=MUTED)
        for sp in ax.spines.values():
            sp.set_color('#2d3748')

        unique_labels = sorted(set(lbl.tolist()))
        handles = []
        for cls in unique_labels:
            m = lbl == cls
            if m.sum() == 0:
                continue
            idx = rng.choice(np.where(m)[0],
                             min(max_pts // len(unique_labels), m.sum()),
                             replace=False)
            col  = CLASS_COLOURS.get(cls, '#ffffff')
            name = CLASS_NAMES.get(cls, f'cls{cls}')
            s    = 1.2 if cls >= 0 else 0.8
            alph = 0.8 if cls >= 0 else 0.4
            ax.scatter(x[idx], y[idx], c=col, s=s,
                       alpha=alph, rasterized=True)
            handles.append(mpatches.Patch(color=col, label=name))
        ax.legend(handles=handles, fontsize=8, facecolor=PANEL,
                  labelcolor=TEXT, edgecolor='#374151',
                  markerscale=5, loc='lower right')

    # Panel 1 — clean stream
    scatter_stream(axes[0],
                   stream_clean.t, stream_clean.x,
                   stream_clean.y, stream_clean.p,
                   labels_clean,
                   f'Clean Stream\n{len(stream_clean):,} events')

    # Panel 2 — weather noise only
    weather_mask = labels_noisy >= 10
    scatter_stream(axes[1],
                   stream_noisy.t[weather_mask],
                   stream_noisy.x[weather_mask],
                   stream_noisy.y[weather_mask],
                   stream_noisy.p[weather_mask],
                   labels_noisy[weather_mask],
                   f'Weather Noise Only\n'
                   f'{weather_mask.sum():,} events '
                   f'({weather_mask.mean()*100:.1f}% of stream)')

    # Panel 3 — combined
    scatter_stream(axes[2],
                   stream_noisy.t, stream_noisy.x,
                   stream_noisy.y, stream_noisy.p,
                   labels_noisy,
                   f'Combined (Objects + Weather)\n'
                   f'{len(stream_noisy):,} total events')

    fig.suptitle(title, color=TEXT, fontsize=14,
                 fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.01, 1, 0.96])

    path = 'plots/weather_noise.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  Plot saved → {path}")
    return path


# ─── Also generate event-rate-over-time comparison ───────────────────────────
def visualise_weather_rate(stream_clean, stream_noisy,
                           labels_noisy):
    """
    Show event rate over time for clean vs noisy stream.
    Rain creates characteristic SPIKE patterns.
    Snow creates a more uniform elevation of the baseline.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os

    BG    = '#0a0e1a'
    PANEL = '#111827'
    TEXT  = '#e2e8f0'
    MUTED = '#9ca3af'

    fig, axes = plt.subplots(2, 1, figsize=(16, 9))
    fig.patch.set_facecolor(BG)

    bins   = np.linspace(stream_noisy.t[0], stream_noisy.t[-1], 80)
    bc     = (bins[:-1] + bins[1:]) / 2 / 1000

    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_xlabel('Time (ms)', color=MUTED, fontsize=10)
        ax.set_ylabel('Events per bin', color=MUTED, fontsize=10)
        for sp in ax.spines.values():
            sp.set_color('#2d3748')

    # Top: clean signal rate
    cnt_c, _ = np.histogram(stream_clean.t, bins=bins)
    axes[0].fill_between(bc, cnt_c, alpha=0.8,
                         color='#22c55e', label='Signal (objects)')
    axes[0].set_title('Event Rate — Clean Stream (objects only)',
                      color=TEXT, fontsize=12, fontweight='bold', pad=8)
    axes[0].legend(fontsize=10, facecolor=PANEL,
                   labelcolor=TEXT, edgecolor='#374151')

    # Bottom: noisy stream broken down by type
    signal_mask = (labels_noisy >= 0) & (labels_noisy < 10)
    rain_mask   = labels_noisy == 10
    snow_mask   = labels_noisy == 11

    cnt_s, _ = np.histogram(stream_noisy.t[signal_mask], bins=bins)
    cnt_r, _ = np.histogram(stream_noisy.t[rain_mask],   bins=bins)
    cnt_sn,_ = np.histogram(stream_noisy.t[snow_mask],   bins=bins)

    axes[1].fill_between(bc, cnt_s,  alpha=0.7,
                         color='#22c55e', label='Signal (objects)')
    axes[1].fill_between(bc, cnt_r,  alpha=0.6,
                         color='#60a5fa', label='Rain noise')
    axes[1].fill_between(bc, cnt_sn, alpha=0.6,
                         color='#e2e8f0', label='Snow noise')
    axes[1].set_title('Event Rate — Noisy Stream (objects + weather)',
                      color=TEXT, fontsize=12, fontweight='bold', pad=8)
    axes[1].legend(fontsize=10, facecolor=PANEL,
                   labelcolor=TEXT, edgecolor='#374151')

    # Annotate: rain spikes vs snow baseline
    if rain_mask.sum() > 0:
        peak_bin = np.argmax(cnt_r)
        axes[1].annotate(
            'Rain burst\n(spike pattern)',
            xy=(bc[peak_bin], cnt_r[peak_bin]),
            xytext=(bc[peak_bin] + 8, cnt_r[peak_bin] + 5),
            color='#60a5fa', fontsize=9,
            arrowprops=dict(arrowstyle='->', color='#60a5fa'))

    if snow_mask.sum() > 0:
        axes[1].annotate(
            'Snow raises\nbaseline uniformly',
            xy=(bc[20], cnt_sn[20]),
            xytext=(bc[20] + 15, cnt_sn[20] + 8),
            color='#e2e8f0', fontsize=9,
            arrowprops=dict(arrowstyle='->', color='#e2e8f0'))

    fig.suptitle('Weather Noise Signature — Rain vs Snow Event Patterns',
                 color=TEXT, fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.01, 1, 0.96])

    path = 'plots/weather_rate.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  Plot saved → {path}")
    return path


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse, os
    from event_reader import EventReader

    parser = argparse.ArgumentParser(
        description='Add rain/snow noise to event stream',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--rain',  type=str, default='medium',
                        choices=['none','light','medium','heavy'])
    parser.add_argument('--snow',  type=str, default='none',
                        choices=['none','light','medium','heavy'])
    parser.add_argument('--n-objects', type=int, default=6)
    args = parser.parse_args()

    rain = None if args.rain == 'none' else args.rain
    snow = None if args.snow == 'none' else args.snow

    os.makedirs('data',  exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    print("Generating clean synthetic scene...")
    stream_clean, labels_clean = EventReader.generate_synthetic(
        n_objects=args.n_objects, seed=42)

    print(f"\nAdding weather noise  (rain={rain}, snow={snow})...")
    wn = WeatherNoise(rain_intensity=rain, snow_intensity=snow)
    stream_noisy, labels_noisy, stats = wn.add_weather(
        stream_clean, labels_clean)

    print(f"\nStream summary:")
    print(f"  Clean  : {len(stream_clean):,} events")
    print(f"  Noisy  : {len(stream_noisy):,} events")
    print(f"  Weather: {stats['total_weather']:,} events "
          f"({stats['weather_pct']:.1f}%)")

    # Save
    stream_noisy.save_npz('data/weather_event_stream.npz')
    np.save('data/weather_labels.npy', labels_noisy)

    print("\nGenerating plots...")
    title = f'Rain={rain or "none"}  |  Snow={snow or "none"}'
    visualise_weather(stream_clean, stream_noisy,
                      labels_clean, labels_noisy, title)
    visualise_weather_rate(stream_clean, stream_noisy, labels_noisy)

    print("\n=== Done ===")
    print("Plots saved:")
    print("  plots/weather_noise.png  — spatial comparison")
    print("  plots/weather_rate.png   — event rate signatures")
    print("\nTo run the full pipeline on the noisy stream:")
    print("  python3 run_pipeline.py --file data/weather_event_stream.npz")
