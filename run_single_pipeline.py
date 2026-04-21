"""
run_single_pipeline.py
======================
Run one specific pipeline configuration and show detailed results.
Edit the CONFIGURATION section below to choose your pipeline.

Usage:
    python run_single_pipeline.py
    python run_single_pipeline.py --detector ADB-Opt --filter GB
    python run_single_pipeline.py --detector Dual-Opt --filter DL
    python run_single_pipeline.py --list
"""
import sys, os, warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn"), argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
import numpy as np
from _quiet import quiet_mode

VERBOSE = '--verbose' in sys.argv
def _run(fn):
    if VERBOSE: return fn()
    with quiet_mode(): return fn()

from event_reader          import EventReader
from noise_filter          import NoiseFilter
from detector              import EventDetector
from detector_adbscan      import ADBScanDetector
from dual_adaptive_detector import DualADBScanDetector
from detector_3d           import ADBScan3DDetector
from weather_filter        import WeatherClusterFilter
from weather_noise         import WeatherNoise
from classifier            import ObjectClassifier, CLASS_NAMES
from dl_weather_filter     import DLWeatherFilter

REAL = set(range(6))
DL_PATH = os.path.join('results', 'models', 'dl_weather_filter.pt')

# ── Available pipelines ───────────────────────────────────────────────────────
DETECTOR_CONFIGS = {
    'DBSCAN': {
        'class': ADBScanDetector,
        'kwargs': dict(window_ms=5, eps_px=10, min_samples=5,
                       adaptive_params=(0.0, 0.0, 10.0),
                       min_track_len=4, eps_mode='radial', verbose=False),
        'dim': '2D', 'k': 'fixed',
        'description': 'Standard DBSCAN, flat eps=10px, k=5 everywhere',
    },
    'ADB-Hard': {
        'class': ADBScanDetector,
        'kwargs': dict(window_ms=5, eps_px=10, min_samples=5,
                       adaptive_params=(0.0001, -0.02, 10.0),
                       min_track_len=4, eps_mode='radial', verbose=False),
        'dim': '2D', 'k': 'ε-only',
        'description': 'ADBScan 2D, hardcoded eps (c2=0.0001, vertex 100px)',
    },
    'ADB-Opt': {
        'class': ADBScanDetector,
        'kwargs': dict(window_ms=5, eps_px=9.5, min_samples=5,
                       adaptive_params=(0.0002, -0.03, 9.5),
                       min_track_len=4, eps_mode='radial', verbose=False),
        'dim': '2D', 'k': 'ε-only',
        'description': 'ADBScan 2D, OPTIMAL eps (c2=0.0002, vertex 75px) — BEST',
    },
    'Dual-Hard': {
        'class': DualADBScanDetector,
        'kwargs': dict(window_ms=5, eps_px=10, min_samples=5,
                       adaptive_params=(0.0001, -0.02, 10.0),
                       k_centre=42.0, k_edge=25.0,
                       min_track_len=4, eps_mode='radial', verbose=False),
        'dim': '2D', 'k': 'ε+k',
        'description': 'Full patent: adaptive eps AND k (hardcoded)',
    },
    'Dual-Opt': {
        'class': DualADBScanDetector,
        'kwargs': dict(window_ms=5, eps_px=9.5, min_samples=5,
                       adaptive_params=(0.0002, -0.03, 9.5),
                       k_centre=42.0, k_edge=25.0,
                       min_track_len=4, eps_mode='radial', verbose=False),
        'dim': '2D', 'k': 'ε+k',
        'description': 'Full patent: adaptive eps AND k (optimal) — True US 10,510,154',
    },
    '3D-DBSCAN': {
        'class': ADBScan3DDetector,
        'kwargs': dict(window_ms=5, eps_px=10, min_samples=5,
                       adaptive_params=(0.0, 0.0, 10.0),
                       min_track_len=4, t_scale_us=500, verbose=False),
        'dim': '3D', 'k': 'fixed',
        'description': 'Standard DBSCAN in (x,y,t) spatiotemporal space',
    },
    '3D-ADB-Opt': {
        'class': ADBScan3DDetector,
        'kwargs': dict(window_ms=5, eps_px=9.5, min_samples=5,
                       adaptive_params=(0.0002, -0.03, 9.5),
                       min_track_len=4, t_scale_us=500, verbose=False),
        'dim': '3D', 'k': 'ε-only',
        'description': 'ADBScan in (x,y,t) space, optimal eps',
    },
}

FILTER_CONFIGS = {
    'none':  {'use_gb': False, 'use_dl': False,
              'description': 'No weather filter — raw detector output'},
    'GB':    {'use_gb': True,  'use_dl': False,
              'description': 'Gradient Boosting on 15 track shape features'},
    'DL':    {'use_gb': False, 'use_dl': True,
              'description': 'EventResNet 1D CNN on raw event sequences'},
    'GB+DL': {'use_gb': True,  'use_dl': True,
              'description': 'DL filter then GB filter (sequential)'},
}

COND_SPECS = [
    ('Clean',       None,     None),
    ('Light Rain',  'light',  None),
    ('Med Rain',    'medium', None),
    ('Heavy Rain',  'heavy',  None),
    ('Light Snow',  None,     'light'),
    ('Med Snow',    None,     'medium'),
    ('Rain+Snow',   'medium', 'light'),
]
COND_WEIGHTS = [1.0, 1.2, 1.5, 1.5, 1.0, 1.2, 1.5]


def list_options():
    print("\nAvailable DETECTORS:")
    for name, cfg in DETECTOR_CONFIGS.items():
        print(f"  {name:<14}  [{cfg['dim']}, k={cfg['k']}]  {cfg['description']}")
    print("\nAvailable FILTERS:")
    for name, cfg in FILTER_CONFIGS.items():
        print(f"  {name:<8}  {cfg['description']}")
    print()


def run_pipeline(det_name, filt_name, verbose=True):
    if det_name not in DETECTOR_CONFIGS:
        print(f"Unknown detector: {det_name}"); list_options(); sys.exit(1)
    if filt_name not in FILTER_CONFIGS:
        print(f"Unknown filter: {filt_name}"); list_options(); sys.exit(1)

    dcfg = DETECTOR_CONFIGS[det_name]
    fcfg = FILTER_CONFIGS[filt_name]

    print("=" * 70)
    print(f"  Pipeline: {det_name} + {filt_name}")
    print(f"  Detector: {dcfg['description']}")
    print(f"  Filter:   {fcfg['description']}")
    print("=" * 70)

    # ── Data ──────────────────────────────────────────────────────────────────
    stream, labels = EventReader.generate_synthetic(n_objects=6, seed=42)
    nf = NoiseFilter(mode='multi_scale', delta_t_us=8000)
    sc_f, kc = nf.filter(stream); lf_c = labels[kc]

    # ── Classifier ────────────────────────────────────────────────────────────
    det0 = EventDetector(window_ms=5, eps_px=10, min_samples=5, verbose=False)
    tr0, ft0 = det0.detect(sc_f, lf_c)
    clf = ObjectClassifier(model_dir=os.path.join('results', 'models'))
    clf.train(ft0, np.array([t.gt_class for t in tr0]),
              augment_factor=5, verbose=False)

    # ── GB filter ─────────────────────────────────────────────────────────────
    gb = None
    if fcfg['use_gb']:
        wn_t = WeatherNoise(rain_intensity='heavy', snow_intensity='medium', seed=42)
        sw_t, lw_t, _ = wn_t.add_weather(stream, labels)
        sf_t, kf_t = nf.filter(sw_t)
        det_w = EventDetector(window_ms=5, eps_px=10, min_samples=5, min_track_len=2, verbose=False)
        tr_w, ft_w = det_w.detect(sf_t, lw_t[kf_t])
        gt_w = np.array([t.gt_class for t in tr_w])
        wm = np.array([g not in REAL for g in gt_w])
        gb = WeatherClusterFilter(mode='gradient_boost', verbose=False)
        gb.train(ft_w[~wm], ft_w[wm], model_type='gradient_boost')

    # ── DL filter ─────────────────────────────────────────────────────────────
    dl = None
    if fcfg['use_dl']:
        dl = DLWeatherFilter(seq_len=256, model_path=DL_PATH, verbose=False)
        if os.path.exists(DL_PATH):
            dl.load()
        else:
            print("  WARNING: DL model not found. Run run_benchmark.py first.")
            dl = None

    # ── Build detector ────────────────────────────────────────────────────────
    det = dcfg['class'](**dcfg['kwargs'])

    # ── Run all conditions ────────────────────────────────────────────────────
    print(f"\n  {'Condition':<16} {'Tracks':>7} {'WeatherRm':>10} "
          f"{'Correct':>8} {'Accuracy':>10}")
    print(f"  {'-'*60}")

    per_cond = []
    for cname, r, s in COND_SPECS:
        sw, lw = (stream, labels) if r is None and s is None else \
            WeatherNoise(rain_intensity=r, snow_intensity=s,
                         seed=123).add_weather(stream, labels)[:2]
        sf, kf = nf.filter(sw)
        lf = lw[kf]

        tr, ft = _run(lambda: det.detect(sf, lf))
        ft15 = ft[:, :15]
        n_raw = len(tr)
        n_removed = 0

        if dl is not None and len(tr) > 0:
            tr_f, _ = dl.filter(tr, sf)
            kept = {id(t) for t in tr_f}
            idxs = [i for i, t2 in enumerate(tr) if id(t2) in kept]
            n_removed += n_raw - len(tr_f)
            tr = tr_f; ft15 = ft15[idxs] if idxs else ft15[:0]

        if gb is not None and len(tr) > 0:
            n_before = len(tr)
            tr, ft15, _, _ = gb.filter(tr, ft15)
            n_removed += n_before - len(tr)

        acc = 0.0; n_correct = 0
        if len(tr) > 0:
            gt = np.array([t.gt_class for t in tr]); v = gt >= 0
            if v.sum() > 0:
                pred, _, _ = clf.predict(ft15[v])
                n_correct = int((pred == gt[v]).sum())
                acc = n_correct / v.sum() * 100

        per_cond.append(acc)
        print(f"  {cname:<16} {n_raw:>7} {n_removed:>10} "
              f"{n_correct:>8} {acc:>9.1f}%")

    wavg = float(np.average(per_cond, weights=COND_WEIGHTS))
    print(f"  {'-'*60}")
    print(f"  {'Weighted Average':<16} {'':>7} {'':>10} "
          f"{'':>8} {wavg:>9.1f}%")
    print(f"\n  RESULT: {det_name} + {filt_name} = {wavg:.1f}%\n")
    return per_cond, wavg


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run single ADBScan pipeline')
    parser.add_argument('--detector', default='ADB-Opt',
                        help='Detector name (default: ADB-Opt)')
    parser.add_argument('--filter',   default='GB',
                        help='Filter name: none/GB/DL/GB+DL (default: GB)')
    parser.add_argument('--list', action='store_true',
                        help='List available options and exit')
    args = parser.parse_args()

    if args.list:
        list_options(); sys.exit(0)

    run_pipeline(args.detector, args.filter)
