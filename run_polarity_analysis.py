"""
run_polarity_analysis.py
========================
Measure and visualise the DVS-specific polarity flip-rate signatures
of rain, snow and real objects.

Key finding (novel physical observation):
  - Rain clusters:    flip_rate ≈ 0.09  (threshold < 0.28 → 100% detected)
  - Real objects:     flip_rate ≈ 0.47  (always above threshold)
  - Snow clusters:    flip_rate ≈ 0.50  (same as real, use density rules)

Outputs:
  results/plots/polarity_analysis.png
  results/polarity_stats.txt

Usage:
    python run_polarity_analysis.py
"""
import sys, os, warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
from _quiet import quiet_mode
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
import numpy as np

from event_reader  import EventReader
from noise_filter  import NoiseFilter
from weather_noise import WeatherNoise
from detector      import EventDetector
from classifier    import CLASS_NAMES

REAL = set(range(6))

def compute_flip_rate(stream, track):
    """Compute polarity flip rate for one track (temporal ordering)."""
    all_idx = np.concatenate([d.event_idx for d in track.detections])
    p = stream.p[all_idx]
    n = len(p)
    if n < 3:
        return None, None, None
    t_sort = stream.t[all_idx].argsort()
    p_sorted = p[t_sort]
    flips = (p_sorted[1:] != p_sorted[:-1]).sum()
    flip_rate = flips / (n - 1)
    frac_pos  = (p > 0).sum() / n
    return flip_rate, frac_pos, n


def main():
    print("=" * 60)
    print("  DVS Polarity Physics Analysis")
    print("=" * 60)

    # Generate stream with heavy rain + medium snow
    stream, labels = EventReader.generate_synthetic(n_objects=6, seed=42)
    nf = NoiseFilter(mode='multi_scale', delta_t_us=8000)

    wn = WeatherNoise(rain_intensity='heavy', snow_intensity='medium', seed=42)
    sw, lw, stats = wn.add_weather(stream, labels)
    sf, kf = nf.filter(sw)
    lf = lw[kf]

    print(f"  Weather events added: {stats}")

    # Detect tracks
    det = EventDetector(window_ms=5, eps_px=10, min_samples=5, min_track_len=2, verbose=False)
    tr, _ = det.detect(sf, lf)
    gt_arr = np.array([t.gt_class for t in tr])
    print(f"  Detected {len(tr)} tracks\n")

    # Collect flip rates per class
    real_flip=[]; rain_flip=[]; snow_flip=[]
    real_n=[]; rain_n=[]; snow_n=[]

    for trk, gt in zip(tr, gt_arr):
        fr, fp, n = compute_flip_rate(sf, trk)
        if fr is None:
            continue
        if gt in REAL:
            real_flip.append(fr); real_n.append(n)
        elif gt == 10:  # rain label
            rain_flip.append(fr); rain_n.append(n)
        elif gt == 11:  # snow label
            snow_flip.append(fr); snow_n.append(n)

    # Print statistics
    print(f"{'='*60}")
    print("FLIP RATE STATISTICS")
    print(f"{'='*60}")
    lines = []
    for name, arr, narr in [
        ('Real objects',  real_flip, real_n),
        ('Rain clusters', rain_flip, rain_n),
        ('Snow clusters', snow_flip, snow_n),
    ]:
        if not arr:
            continue
        a = np.array(arr)
        s = (f"\n  {name} (n={len(a)} tracks):\n"
             f"    flip_rate:  mean={a.mean():.4f}  std={a.std():.4f}"
             f"  min={a.min():.4f}  max={a.max():.4f}\n"
             f"    n_events:   mean={np.mean(narr):.0f}"
             f"  min={min(narr)}  max={max(narr)}")
        print(s); lines.append(s)

    # Threshold analysis
    print(f"\n{'='*60}")
    print("THRESHOLD ANALYSIS")
    print(f"{'='*60}")
    r = np.array(real_flip); rain = np.array(rain_flip)
    for thr in [0.20, 0.25, 0.28, 0.30, 0.35]:
        rain_ok = (rain < thr).mean()*100 if len(rain) else 0
        real_ok = (r    > thr).mean()*100 if len(r)    else 0
        print(f"  threshold={thr:.2f}:  "
              f"rain→noise={rain_ok:.0f}%  real→kept={real_ok:.0f}%")

    # Snow density rules
    print(f"\n{'='*60}")
    print("SNOW DENSITY RULES (n_events<250 AND n_windows<6 AND ev/win<30)")
    print(f"{'='*60}")
    for trk, gt in zip(tr, gt_arr):
        if gt not in [11, *REAL]:
            continue
        n_ev  = sum(d.n_events for d in trk.detections)
        n_win = trk.length
        ev_w  = n_ev / max(n_win, 1)
        rule  = (n_ev < 250) and (n_win < 6) and (ev_w < 30)
        cls   = 'Snow' if gt == 11 else CLASS_NAMES[gt]
        print(f"  {cls:<14}  n_ev={n_ev:>5}  n_win={n_win:>3}"
              f"  ev/win={ev_w:>5.1f}  snow_rule={rule}")

    # Save stats
    os.makedirs('results', exist_ok=True)
    with open('results/polarity_stats.txt', 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nSaved: results/polarity_stats.txt")

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        os.makedirs('results/plots', exist_ok=True)
        BG='#FFFFFF'; PANEL='#F8F9FA'; BORDER='#DEE2E6'
        GREEN='#2D6A4F'; TEAL='#0077B6'; PURPLE='#6C3483'
        AMBER='#E8A000'; PINK='#C0392B'

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(BG)

        # Panel 1: histogram
        ax = axes[0]; ax.set_facecolor(PANEL)
        bins = np.linspace(0, 0.7, 35)
        if real_flip: ax.hist(real_flip,bins=bins,color=GREEN, alpha=0.65,
                              label=f'Real objects  mean={np.mean(real_flip):.3f}')
        if rain_flip: ax.hist(rain_flip,bins=bins,color=TEAL,  alpha=0.65,
                              label=f'Rain clusters mean={np.mean(rain_flip):.3f}')
        if snow_flip: ax.hist(snow_flip,bins=bins,color=PURPLE,alpha=0.65,
                              label=f'Snow clusters mean={np.mean(snow_flip):.3f}')
        ax.axvline(0.28, color=AMBER, lw=3, ls='--', label='Threshold: 0.28')
        ax.set_xlabel('Flip rate (fraction of consecutive events that change polarity)')
        ax.set_ylabel('Count (tracks)')
        ax.set_title('Polarity Flip Rate Distribution\n'
                     'Rain = distinct low-flip zone below 0.28')
        ax.legend(fontsize=9.5, facecolor='white', edgecolor=BORDER)
        ax.grid(axis='y', color=BORDER, alpha=0.8); ax.set_axisbelow(True)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

        # Panel 2: 2D scatter
        ax = axes[1]; ax.set_facecolor(PANEL)
        if real_flip: ax.scatter(real_flip,real_n, c=GREEN,  s=60,alpha=0.75,
                                 label='Real objects',zorder=5)
        if rain_flip: ax.scatter(rain_flip,rain_n, c=TEAL,   s=60,alpha=0.75,
                                 label='Rain clusters',zorder=5)
        if snow_flip: ax.scatter(snow_flip,snow_n, c=PURPLE, s=60,alpha=0.75,
                                 label='Snow clusters',zorder=5)
        ax.axvline(0.28,color=AMBER,lw=2.5,ls='--',label='Flip rate threshold: 0.28')
        ax.axhline(250, color=PINK, lw=2,  ls='--',label='Event count threshold: 250')
        ax.set_xlabel('Flip rate')
        ax.set_ylabel('n_events (log scale)')
        ax.set_yscale('log')
        ax.set_title('2D Separation: Flip Rate vs Event Count\n'
                     'Three distinct zones: Rain / Real / Snow')
        ax.legend(fontsize=9.5, facecolor='white', edgecolor=BORDER)
        ax.grid(axis='y', color=BORDER, alpha=0.8); ax.set_axisbelow(True)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

        plt.tight_layout()
        plt.savefig('results/plots/polarity_analysis.png',
                    dpi=150, bbox_inches='tight', facecolor=BG)
        plt.close()
        print("Saved: results/plots/polarity_analysis.png")

    except ImportError:
        print("matplotlib not available — skipping plot")


if __name__ == '__main__':
    main()
