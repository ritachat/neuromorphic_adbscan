"""
run_benchmark.py — Master benchmark, ALL 28 pipeline combinations.
Usage:
    python run_benchmark.py            # quiet (clean progress bar)
    python run_benchmark.py --verbose  # show all internal output
"""
import sys, os, pickle, time, argparse, warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()
VERBOSE = args.verbose

import numpy as np
from _quiet import quiet_mode
from event_reader           import EventReader
from noise_filter           import NoiseFilter
from detector               import EventDetector
from detector_adbscan       import ADBScanDetector
from dual_adaptive_detector import DualADBScanDetector
from detector_3d            import ADBScan3DDetector
from weather_filter         import WeatherClusterFilter
from weather_noise          import WeatherNoise
from classifier             import ObjectClassifier
from dl_weather_filter      import DLWeatherFilter

REAL         = set(range(6))
DL_PATH      = os.path.join('results', 'models', 'dl_weather_filter.pt')
COND_NAMES   = ['Clean','Light Rain','Med Rain','Heavy Rain','Light Snow','Med Snow','Rain+Snow']
COND_WEIGHTS = [1.0, 1.2, 1.5, 1.5, 1.0, 1.2, 1.5]

def _run(fn):
    if VERBOSE: return fn()
    with quiet_mode(): return fn()

def done(msg): print(f"  {msg}")

def setup_all():
    print("=" * 65)
    print("  Neuromorphic ADBScan — Master Benchmark")
    print("  US Patent 10,510,154 — Extension to DVS DAVIS346")
    print("=" * 65)
    t0 = time.time()
    os.makedirs('results/models', exist_ok=True)

    print("  [1/5] Generating stream + noise filter...", end=' ', flush=True)
    stream, labels = _run(lambda: EventReader.generate_synthetic(n_objects=6, seed=42))
    nf = NoiseFilter(mode='multi_scale', delta_t_us=8000, verbose=VERBOSE)
    sc_f, kc = _run(lambda: nf.filter(stream))
    lf_c = labels[kc]
    done(f"{len(sc_f.t):,} events ({len(stream.t):,} raw)")

    print("  [2/5] Training object classifier...", end=' ', flush=True)
    det0 = EventDetector(window_ms=5, eps_px=10, min_samples=5, verbose=False)
    tr0, ft0 = _run(lambda: det0.detect(sc_f, lf_c))
    clf = ObjectClassifier(model_dir=os.path.join('results', 'models'))
    _run(lambda: clf.train(ft0, np.array([t.gt_class for t in tr0]),
                           augment_factor=5, verbose=False))
    done(f"{len(tr0)} tracks")

    print("  [3/5] Training GB weather filter...", end=' ', flush=True)
    wn_t = WeatherNoise(rain_intensity='heavy', snow_intensity='medium', seed=42)
    wn_t.verbose = False
    sw_t, lw_t, _ = _run(lambda: wn_t.add_weather(stream, labels))
    sf_t, kf_t = _run(lambda: nf.filter(sw_t))
    det_w = EventDetector(window_ms=5, eps_px=10, min_samples=5, min_track_len=2, verbose=False)
    tr_w, ft_w = _run(lambda: det_w.detect(sf_t, lw_t[kf_t]))
    gt_w = np.array([t.gt_class for t in tr_w])
    wm = np.array([g not in REAL for g in gt_w])
    gb = WeatherClusterFilter(mode='gradient_boost', verbose=False)
    _run(lambda: gb.train(ft_w[~wm], ft_w[wm], model_type='gradient_boost'))
    done(f"{wm.sum()} weather / {(~wm).sum()} real tracks")

    dl = None
    if os.path.exists(DL_PATH):
        print("  [4/5] Loading DL filter...", end=' ', flush=True)
        dl = DLWeatherFilter(seq_len=256, model_path=DL_PATH, verbose=False)
        _run(dl.load); done(DL_PATH)
    else:
        done(f"[4/5] DL model not found — run: python run_train_dl.py  (skipping +DL)")

    print("  [5/5] Pre-loading 7 weather scenes...", end=' ', flush=True)
    scenes = []
    for cname, r, s in [
        ('Clean',None,None),('Light Rain','light',None),('Med Rain','medium',None),
        ('Heavy Rain','heavy',None),('Light Snow',None,'light'),
        ('Med Snow',None,'medium'),('Rain+Snow','medium','light'),
    ]:
        wn2 = WeatherNoise(rain_intensity=r, snow_intensity=s, seed=123)
        wn2.verbose = False
        if r is None and s is None:
            sw2, lw2 = stream, labels
        else:
            sw2, lw2, _ = _run(lambda wn2=wn2: wn2.add_weather(stream, labels))
        sf2, kf2 = _run(lambda sw2=sw2: nf.filter(sw2))
        scenes.append((cname, sf2, lw2[kf2]))
    done(f"{len(scenes)} scenes ready")
    print(f"\n  Setup complete in {time.time()-t0:.0f}s\n")
    return stream, labels, nf, clf, gb, dl, scenes


def score_det(det, sf, lf, clf, gb, dl, use_gb, use_dl):
    def _inner():
        tr, ft = det.detect(sf, lf)
        ft15 = ft[:, :15]
        if use_dl and dl is not None and len(tr) > 0:
            tr_f, _ = dl.filter(tr, sf)
            kept = {id(t) for t in tr_f}
            idxs = [i for i, t2 in enumerate(tr) if id(t2) in kept]
            tr = tr_f; ft15 = ft15[idxs] if idxs else ft15[:0]
        if use_gb and len(tr) > 0:
            tr, ft15, _, _ = gb.filter(tr, ft15)
        if len(tr) == 0: return 0.0
        gt = np.array([t.gt_class for t in tr]); v = gt >= 0
        if v.sum() == 0: return 0.0
        pred, _, _ = clf.predict(ft15[v])
        return (pred == gt[v]).mean() * 100
    return _run(_inner)


def run_benchmark(stream, labels, nf, clf, gb, dl, scenes):
    K2 = dict(min_track_len=4, eps_mode='radial', verbose=False)
    K3 = dict(min_track_len=4, t_scale_us=500, verbose=False)
    detectors = [
        ("DBSCAN",    ADBScanDetector(window_ms=5,eps_px=10,min_samples=5,adaptive_params=(0.0,0.0,10.0),**K2),"2D","fixed"),
        ("ADB-Hard",  ADBScanDetector(window_ms=5,eps_px=10,min_samples=5,adaptive_params=(0.0001,-0.02,10.0),**K2),"2D","ε-only"),
        ("ADB-Opt",   ADBScanDetector(window_ms=5,eps_px=9.5,min_samples=5,adaptive_params=(0.0002,-0.03,9.5),**K2),"2D","ε-only"),
        ("Dual-Hard", DualADBScanDetector(window_ms=5,eps_px=10,min_samples=5,adaptive_params=(0.0001,-0.02,10.0),k_centre=42.0,k_edge=25.0,**K2),"2D","ε+k"),
        ("Dual-Opt",  DualADBScanDetector(window_ms=5,eps_px=9.5,min_samples=5,adaptive_params=(0.0002,-0.03,9.5),k_centre=42.0,k_edge=25.0,**K2),"2D","ε+k"),
        ("3D-DBSCAN", ADBScan3DDetector(window_ms=5,eps_px=10,min_samples=5,adaptive_params=(0.0,0.0,10.0),**K3),"3D","fixed"),
        ("3D-ADB-Opt",ADBScan3DDetector(window_ms=5,eps_px=9.5,min_samples=5,adaptive_params=(0.0002,-0.03,9.5),**K3),"3D","ε-only"),
    ]
    filters = [("no filter",False,False),("+GB",True,False),("+DL",False,True),("+GB+DL",True,True)]

    results = {}; best_avg = -1; n_done = 0; n_total = len(detectors)*len(filters)
    print("=" * 100)
    print(f"{'Method':<22} {'Dim':>4} {'k':>7} {'Filter':>7}  {'Cl':>5} {'LtR':>5} {'MdR':>5} {'HvR':>5} {'LtS':>5} {'MdS':>5} {'R+S':>5}  {'WAvg':>7}")
    print("-" * 100)

    prev_filt = ""
    for filt_name, use_gb, use_dl in filters:
        if use_dl and dl is None:
            print(f"\n  [Skipping {filt_name} — DL model not available]\n")
            n_done += len(detectors); continue
        if filt_name != prev_filt and prev_filt: print()
        prev_filt = filt_name

        for dname, det, dim, ktype in detectors:
            n_done += 1
            if not VERBOSE:
                print(f"  [{n_done:>2}/{n_total}] {dname} {filt_name}...", end='\r', flush=True)
            t0 = time.time()
            sc  = [score_det(det,sf,lf,clf,gb,dl,use_gb,use_dl) for _,sf,lf in scenes]
            avg = float(np.average(sc, weights=COND_WEIGHTS))
            key = f"{dname} {filt_name}"
            results[key] = {'scores':sc,'avg':avg,'dim':dim,'k':ktype,'filter':filt_name,'det':dname,'time_s':time.time()-t0}
            if avg > best_avg: best_avg = avg
            mark = " ★" if avg == best_avg else ""
            vals = "  ".join(f"{v:>4.0f}%" for v in sc)
            print(f"{dname:<22} {dim:>4} {ktype:>7} {filt_name:>7}  {vals}  {avg:>6.1f}%{mark}")
    return results


def print_summary(results):
    ranked = sorted(results.items(), key=lambda x: -x[1]['avg'])
    print("\n" + "=" * 80)
    print("OVERALL RANKING")
    print("=" * 80)
    for rank, (key, info) in enumerate(ranked, 1):
        vals = "  ".join(f"{v:>3.0f}%" for v in info['scores'])
        print(f"  {rank:>2}. {key:<32}  {vals}  avg={info['avg']:.1f}%")

    print("\n" + "=" * 80)
    print("BEST FILTER PER DETECTOR")
    print("=" * 80)
    seen = {}
    for key, info in ranked:
        d = info['det']
        if d not in seen: seen[d] = (key, info['avg'], info['filter'])
    for d, (key, avg, filt) in seen.items():
        print(f"  {d:<14}  best={avg:.1f}%  filter={filt:<8}  ({key})")

    print("\n" + "=" * 80)
    print("BEST METHOD PER CONDITION")
    print("=" * 80)
    for i, cname in enumerate(COND_NAMES):
        best_k = max(results, key=lambda k: results[k]['scores'][i])
        print(f"  {cname:<14}: {best_k:<35}  ({results[best_k]['scores'][i]:.0f}%)")

    bk, bi = ranked[0]
    print(f"\n{'='*80}")
    print(f"  ★  BEST PIPELINE: {bk}  →  {bi['avg']:.1f}%")
    print(f"     {bi['dim']} clustering  |  k={bi['k']}  |  filter={bi['filter']}")
    print(f"{'='*80}\n")


def plot_results(results):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    os.makedirs('results/plots', exist_ok=True)
    ranked = sorted(results.items(), key=lambda x: -x[1]['avg'])
    labels = [k for k,_ in ranked]; avgs = [v['avg'] for _,v in ranked]
    cmap = {'DBSCAN':'#495057','ADB':'#E8A000','Dual':'#2D6A4F','3D-D':'#1ABC9C','3D-A':'#C0392B'}
    cols = [next((c for p,c in cmap.items() if k.startswith(p)), '#6C3483') for k in labels]

    fig, ax = plt.subplots(figsize=(14,9)); fig.patch.set_facecolor('white'); ax.set_facecolor('#F8F9FA')
    bars = ax.barh(labels, avgs, color=cols, alpha=0.85)
    for bar,v in zip(bars,avgs):
        ax.text(v+0.1,bar.get_y()+bar.get_height()/2,f'{v:.1f}%',va='center',fontsize=9,fontweight='bold')
    ax.set_xlim(48,96); ax.invert_yaxis()
    ax.axvline(64.6,color='grey',lw=1.5,ls='--',alpha=0.5,label='No-filter baseline 64.6%')
    ax.set_xlabel('Weighted average accuracy (7 conditions, severity-weighted)')
    ax.set_title('All 28 Pipeline Combinations — Ranked\nColours: Amber=ADB-Opt  Green=Dual-Opt  Grey=DBSCAN  Teal/Red=3D',fontsize=11,fontweight='bold')
    ax.legend(fontsize=9); ax.grid(axis='x',color='#DEE2E6',alpha=0.8); ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_edgecolor('#DEE2E6')
    plt.tight_layout(); plt.savefig('results/plots/ranking.png',dpi=150,bbox_inches='tight'); plt.close()

    top_keys = [k for k,_ in ranked if 'no filter' not in k][:12]
    sc_mat = np.array([results[k]['scores'] for k in top_keys])
    fig, ax = plt.subplots(figsize=(14,6)); fig.patch.set_facecolor('white')
    im = ax.imshow(sc_mat, cmap='RdYlGn', vmin=60, vmax=100, aspect='auto')
    ax.set_xticks(range(len(COND_NAMES))); ax.set_xticklabels(COND_NAMES,fontsize=10)
    ax.set_yticks(range(len(top_keys))); ax.set_yticklabels(top_keys,fontsize=9)
    for i in range(len(top_keys)):
        for j in range(len(COND_NAMES)):
            ax.text(j,i,f'{sc_mat[i,j]:.0f}%',ha='center',va='center',fontsize=8,fontweight='bold',
                    color='black' if sc_mat[i,j]>72 else 'white')
    plt.colorbar(im,ax=ax,label='Accuracy (%)')
    ax.set_title('Per-Condition Accuracy Heatmap — Top 12 Pipelines',fontsize=11,fontweight='bold')
    plt.tight_layout(); plt.savefig('results/plots/heatmap.png',dpi=150,bbox_inches='tight'); plt.close()
    print(f"  Plots → results/plots/ranking.png  results/plots/heatmap.png")


if __name__ == '__main__':
    t_start = time.time()
    stream, labels, nf, clf, gb, dl, scenes = setup_all()
    results = run_benchmark(stream, labels, nf, clf, gb, dl, scenes)
    print_summary(results)
    os.makedirs('results', exist_ok=True)
    with open('results/benchmark_results.pkl','wb') as f:
        pickle.dump({'results':results,'cond_names':COND_NAMES,'cond_weights':COND_WEIGHTS},f)
    print(f"  Results → results/benchmark_results.pkl")
    plot_results(results)
    print(f"\n  Total runtime: {time.time()-t_start:.0f}s\n")
