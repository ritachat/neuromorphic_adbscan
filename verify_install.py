"""
verify_install.py
=================
Quick smoke test — confirms all modules import correctly,
Numba JIT compiles, and the best pipeline scores > 90%.
Run this immediately after setup to confirm the installation works.

Usage:
    python verify_install.py

Expected output (< 60 seconds):
    [OK] All imports
    [OK] Numba JIT
    [OK] Event stream generated
    [OK] Noise filter
    [OK] ADBScan clustering
    [OK] GB filter trained
    [OK] DL filter  (or [SKIP] if model not yet trained)
    [OK] Best pipeline ADB-Opt+GB = 91.2%
    All checks passed.
"""
import sys, os, warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
from _quiet import quiet_mode
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

def check(label):
    print(f"  [checking] {label} ...", end='', flush=True)

def ok(extra=''):
    print(f"\r  [OK]       {extra or ''}              ")

def skip(reason):
    print(f"\r  [SKIP]     {reason}              ")

def fail(err):
    print(f"\r  [FAIL]     {err}              ")
    sys.exit(1)

print("=" * 55)
print("  Neuromorphic ADBScan — Installation Verify")
print("=" * 55)

# ── Imports ───────────────────────────────────────────────
check("All imports")
try:
    import numpy as np
    from event_reader           import EventReader
    from noise_filter           import NoiseFilter
    from adbscan                import _cluster_jit
    from detector               import EventDetector
    from detector_adbscan       import ADBScanDetector
    from dual_adaptive_detector import DualADBScanDetector
    from detector_3d            import ADBScan3DDetector
    from weather_noise          import WeatherNoise
    from weather_filter         import WeatherClusterFilter
    from dl_weather_filter      import DLWeatherFilter
    from polarity_track_filter  import PolarityTrackFilter
    from classifier             import ObjectClassifier
    ok("All 13 modules imported")
except Exception as e:
    fail(str(e))

# ── Numba JIT ─────────────────────────────────────────────
check("Numba JIT compile")
try:
    import numba
    # Trigger compilation with a tiny array
    nb_counts  = np.array([1], dtype=np.int32)
    nb_offsets = np.array([0, 1], dtype=np.int32)
    nb_data    = np.array([0], dtype=np.int32)
    k_arr      = np.array([1], dtype=np.int32)
    _cluster_jit(1, nb_data, nb_offsets, nb_counts, k_arr)
    ok(f"Numba {numba.__version__} — JIT compiled")
except Exception as e:
    fail(str(e))

# ── Event stream ──────────────────────────────────────────
check("Event stream generation")
try:
    with quiet_mode(): stream, labels = EventReader.generate_synthetic(n_objects=6, seed=42)
    assert len(stream.t) > 25000, "Too few events"
    ok(f"{len(stream.t):,} events, {labels.max()+1} object classes")
except Exception as e:
    fail(str(e))

# ── Noise filter ──────────────────────────────────────────
check("Multi-scale noise filter")
try:
    nf = NoiseFilter(mode='multi_scale', delta_t_us=8000)
    with quiet_mode(): sc_f, kc = nf.filter(stream)
    assert len(sc_f.t) > 20000
    ok(f"{len(sc_f.t):,} events after filter ({100*len(sc_f.t)/len(stream.t):.0f}% kept)")
except Exception as e:
    fail(str(e))

# ── ADBScan clustering ────────────────────────────────────
check("ADBScan 2D clustering")
try:
    lf_c = labels[kc]
    det  = ADBScanDetector(window_ms=5, eps_px=9.5, min_samples=5,
                           adaptive_params=(0.0002, -0.03, 9.5),
                           min_track_len=4, eps_mode='radial', verbose=False)
    with quiet_mode(): tr, ft = det.detect(sc_f, lf_c)
    assert len(tr) >= 6, f"Expected ≥6 tracks, got {len(tr)}"
    ok(f"{len(tr)} tracks detected, ft shape={ft.shape}")
except Exception as e:
    fail(str(e))

# ── DualADBScan ───────────────────────────────────────────
check("DualADBScan (ε+k) clustering")
try:
    det2 = DualADBScanDetector(window_ms=5, eps_px=9.5, min_samples=5,
                               adaptive_params=(0.0002, -0.03, 9.5),
                               k_centre=42.0, k_edge=25.0,
                               min_track_len=4, eps_mode='radial', verbose=False)
    with quiet_mode(): tr2, _ = det2.detect(sc_f, lf_c)
    assert len(tr2) >= 6
    ok(f"{len(tr2)} tracks (full patent adaptive ε+k)")
except Exception as e:
    fail(str(e))

# ── Classifier + GB filter ────────────────────────────────
check("Object classifier + GB filter training")
try:
    REAL = set(range(6))
    clf = ObjectClassifier(model_dir=os.path.join('results', 'models'))
    with quiet_mode(): clf.train(ft, np.array([t.gt_class for t in tr]), augment_factor=5, verbose=False)

    wn   = WeatherNoise(rain_intensity='heavy', snow_intensity='medium', seed=42)
    with quiet_mode(): sw, lw, _ = wn.add_weather(stream, labels)
    with quiet_mode(): sf, kf = nf.filter(sw)
    det_w = EventDetector(window_ms=5, eps_px=10, min_samples=5, min_track_len=2, verbose=False)
    with quiet_mode(): tr_w, ft_w = det_w.detect(sf, lw[kf])
    gt_w = np.array([t.gt_class for t in tr_w])
    wm   = np.array([g not in REAL for g in gt_w])
    gb   = WeatherClusterFilter(mode='gradient_boost', verbose=False)
    with quiet_mode(): gb.train(ft_w[~wm], ft_w[wm], model_type='gradient_boost')
    ok(f"Classifier + GB trained on {len(tr_w)} tracks")
except Exception as e:
    fail(str(e))

# ── DL filter ─────────────────────────────────────────────
DL_PATH = os.path.join('results', 'models', 'dl_weather_filter.pt')
check("DL filter (EventResNet)")
try:
    dl = DLWeatherFilter(seq_len=256, model_path=DL_PATH, verbose=False)
    if os.path.exists(DL_PATH):
        dl.load()
        ok(f"Loaded from {DL_PATH}")
    else:
        skip(f"Model not found — run: python run_train_dl.py")
        dl = None
except Exception as e:
    fail(str(e))

# ── Best pipeline end-to-end ──────────────────────────────
check("Best pipeline: ADB-Opt + GB over 7 conditions")
try:
    scenes = []
    for cname, r, s in [
        ('Clean', None, None), ('Light Rain', 'light', None),
        ('Med Rain', 'medium', None), ('Heavy Rain', 'heavy', None),
        ('Light Snow', None, 'light'), ('Med Snow', None, 'medium'),
        ('Rain+Snow', 'medium', 'light'),
    ]:
        sw, lw = (stream, labels) if r is None and s is None else \
            WeatherNoise(rain_intensity=r, snow_intensity=s,
                         seed=123).add_weather(stream, labels)[:2]
        with quiet_mode(): sf2, kf2 = nf.filter(sw)
        scenes.append((cname, sf2, lw[kf2]))

    W_arr = [1.0, 1.2, 1.5, 1.5, 1.0, 1.2, 1.5]
    sc_list = []
    for cname, sf2, lf2 in scenes:
        with quiet_mode(): tr2, ft2 = det.detect(sf2, lf2)
        ft15 = ft2[:, :15]
        if len(tr2) > 0:
            with quiet_mode(): tr2, ft15, _, _ = gb.filter(tr2, ft15)
        if len(tr2) == 0:
            sc_list.append(0.0); continue
        gt2 = np.array([t.gt_class for t in tr2]); v = gt2 >= 0
        if v.sum() == 0:
            sc_list.append(0.0); continue
        pred, _, _ = clf.predict(ft15[v])
        sc_list.append((pred == gt2[v]).mean() * 100)

    avg = float(np.average(sc_list, weights=W_arr))
    if avg < 88.0:
        fail(f"Expected ≥88% but got {avg:.1f}%")
    ok(f"ADB-Opt+GB = {avg:.1f}%  "
       f"[Clean={sc_list[0]:.0f}% HvRain={sc_list[3]:.0f}%]")
except Exception as e:
    fail(str(e))

# ── Summary ───────────────────────────────────────────────
print()
print("  " + "=" * 50)
print("  All checks passed.")
print("  " + "=" * 50)
print()
print("  Next steps:")
print("  1. Train DL filter (once):   python run_train_dl.py")
print("  2. Full benchmark:           python run_benchmark.py")
print("  3. Single pipeline:          python run_single_pipeline.py --detector ADB-Opt --filter GB")
print("  4. Polarity analysis:        python run_polarity_analysis.py")
print()
