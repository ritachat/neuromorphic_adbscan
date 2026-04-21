"""
run_train_dl.py
===============
Train the EventResNet DL weather filter and save the model.
Run this ONCE before running run_benchmark.py or selecting the DL filter.

The DL filter (EventResNet) is a 1D ResNet that reads raw event sequences
(x, y, t, polarity) and classifies each track as real object / rain / snow.

Model saved to: results/models/dl_weather_filter.pt
Training time:  ~3–5 minutes on CPU

Usage:
    python run_train_dl.py
"""
import sys, os, warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
from _quiet import quiet_mode, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
import numpy as np

from event_reader      import EventReader
from noise_filter      import NoiseFilter
from detector          import EventDetector
from weather_noise     import WeatherNoise
from dl_weather_filter import DLWeatherFilter

REAL    = set(range(6))
DL_PATH = os.path.join('results', 'models', 'dl_weather_filter.pt')


def main():
    print("=" * 60)
    print("  Training EventResNet DL Weather Filter")
    print("=" * 60)
    t0 = time.time()
    os.makedirs('results/models', exist_ok=True)

    # Clean data
    stream, labels = EventReader.generate_synthetic(n_objects=6, seed=42)
    nf = NoiseFilter(mode='multi_scale', delta_t_us=8000)
    sc_f, kc = nf.filter(stream); lf_c = labels[kc]

    det_clean = EventDetector(window_ms=5, eps_px=10, min_samples=5, min_track_len=2, verbose=False)
    tr_clean, _ = det_clean.detect(sc_f, lf_c)
    gt_clean = np.array([t.gt_class for t in tr_clean])
    print(f"  Clean tracks: {len(tr_clean)}")

    # Weather data
    wn = WeatherNoise(rain_intensity='heavy', snow_intensity='medium', seed=99)
    sw, lw, stats = wn.add_weather(stream, labels)
    sf, kf = nf.filter(sw); lf = lw[kf]

    det_w = EventDetector(window_ms=5, eps_px=10, min_samples=5, min_track_len=2, verbose=False)
    tr_w, _ = det_w.detect(sf, lf)
    gt_w = np.array([t.gt_class for t in tr_w])
    print(f"  Weather tracks: {len(tr_w)}  {stats}")

    # Train
    dl = DLWeatherFilter(seq_len=256, epochs=50, model_path=DL_PATH, verbose=True)
    dl.train(tr_clean, sc_f, tr_w, sf, gt_clean, gt_w)

    print(f"\n  Model saved to: {DL_PATH}")
    print(f"  Training time:  {time.time()-t0:.0f}s")
    print("\n  You can now run:  python run_benchmark.py")
    print("  Or select DL filter in: python run_single_pipeline.py --filter DL")


if __name__ == '__main__':
    main()
