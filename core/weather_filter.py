"""
weather_filter.py
=================
Post-DBSCAN weather cluster filter.

Removes rain and snow clusters BEFORE the object classifier runs,
so the MLP never sees weather noise — only real objects.

Two approaches implemented:

APPROACH 1 — Rule-Based Filter
--------------------------------
Uses physical characteristics of rain/snow to reject clusters:

  Rain streaks are:
    - Narrow width  (1–5 px)
    - Taller than wide (aspect ratio < 0.5)
    - Short duration  (< 8ms)
    - High polarity separation (clear +1 leading, -1 trailing edge)
    - High vertical velocity  (falling fast)
    - Small area  (< 100 px²)

  Snow blobs are:
    - Roughly circular (aspect ratio 0.6–1.4)
    - Small radius  (< 15 px)
    - Slow speed  (< 1.5 px/ms)
    - Low event density  (diffuse)
    - Short lifetime  (< 20ms)

  Real objects are:
    - Larger bounding boxes
    - Longer lifetimes  (persist across many windows)
    - Consistent motion direction
    - Higher event density

APPROACH 2 — Logistic Regression Classifier
---------------------------------------------
Train a binary classifier:
    Input  : 15-dim feature vector per cluster
    Output : 0 = real object   1 = weather noise

Training data is generated synthetically:
    Positive (weather) examples — from weather_noise.py clusters
    Negative (object)  examples — from clean synthetic clusters

The classifier learns the decision boundary automatically from data,
without hand-tuning thresholds.

Usage:
    from weather_filter import WeatherClusterFilter

    # Rule-based (no training needed)
    wf = WeatherClusterFilter(mode='rule_based')
    clean_tracks, weather_tracks = wf.filter(tracks, features)

    # ML-based (trains automatically on first call)
    wf = WeatherClusterFilter(mode='ml')
    clean_tracks, weather_tracks = wf.filter(tracks, features)
"""

import numpy as np
import pickle
import os
from dataclasses import dataclass
from typing import List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# Feature indices (must match detector.py feature order)
F_WIDTH      = 0
F_HEIGHT     = 1
F_ASPECT     = 2
F_N_EVENTS   = 3
F_DURATION   = 4
F_SPEED      = 5
F_SPEED_X    = 6
F_SPEED_Y    = 7
F_DIRECTION  = 8
F_POL_RATIO  = 9
F_DENSITY    = 10
F_CX         = 11
F_CY         = 12
F_STRAIGHT   = 13
F_OSC        = 14

FEATURE_NAMES = [
    'Width', 'Height', 'Aspect', 'N_events', 'Duration_ms',
    'Speed', 'Speed_X', 'Speed_Y', 'Direction', 'Pol_ratio',
    'Density', 'CX_norm', 'CY_norm', 'Straightness', 'Osc_amp'
]


# ─────────────────────────────────────────────────────────────────────────────
class WeatherClusterFilter:
    """
    Post-DBSCAN filter to remove rain/snow clusters before classification.

    Parameters
    ----------
    mode : str
        'rule_based'  — physics-based thresholds (no training needed)
        'logistic'    — Logistic Regression binary classifier
        'gradient_boost' — Gradient Boosting classifier (more powerful)
        'both'        — rule_based AND logistic must agree (most conservative)
    model_path : str
        Where to save/load the trained ML model.
    verbose : bool
        Print filtering details.
    """

    MODES = ('rule_based', 'logistic', 'gradient_boost', 'both')

    def __init__(self, mode='logistic', model_path='models/weather_filter.pkl',
                 verbose=True):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.mode       = mode
        self.model_path = model_path
        self.verbose    = verbose
        self.scaler     = StandardScaler()
        self.clf        = None
        self._trained   = False

    # ── Public API ────────────────────────────────────────────────────────────

    def filter(self, tracks, features: np.ndarray):
        """
        Filter weather clusters from a list of tracks.

        Parameters
        ----------
        tracks   : List[Track]       from EventDetector
        features : np.ndarray (N,15) feature matrix

        Returns
        -------
        clean_tracks   : List[Track]   real object tracks
        clean_features : np.ndarray    features for clean tracks
        weather_tracks : List[Track]   rejected weather clusters
        weather_mask   : np.ndarray    bool, True = weather noise
        """
        if len(tracks) == 0:
            return [], np.zeros((0, 15)), [], np.array([], dtype=bool)

        if self.mode == 'rule_based':
            weather_mask = self._rule_based(features)

        elif self.mode == 'logistic':
            weather_mask = self._ml_filter(features, model_type='logistic')

        elif self.mode == 'gradient_boost':
            weather_mask = self._ml_filter(features, model_type='gradient_boost')

        elif self.mode == 'both':
            # Both filters must agree for rejection (conservative)
            rules = self._rule_based(features)
            ml    = self._ml_filter(features, model_type='logistic')
            weather_mask = rules & ml

        clean_idx   = np.where(~weather_mask)[0]
        weather_idx = np.where( weather_mask)[0]

        clean_tracks   = [tracks[i]   for i in clean_idx]
        weather_tracks = [tracks[i]   for i in weather_idx]
        clean_features = features[clean_idx]

        if self.verbose:
            print(f"\n[WeatherFilter] Mode: {self.mode}")
            print(f"  Input tracks  : {len(tracks)}")
            print(f"  Clean objects : {len(clean_tracks)}")
            print(f"  Weather noise : {len(weather_tracks)}")
            if len(weather_tracks) > 0:
                print(f"  Rejected tracks: {list(weather_idx)}")

        return clean_tracks, clean_features, weather_tracks, weather_mask

    def train(self, clean_features: np.ndarray,
              weather_features: np.ndarray,
              model_type: str = 'logistic'):
        """
        Train the ML weather filter on labelled feature vectors.

        Parameters
        ----------
        clean_features   : (N, 15) features of real object clusters
        weather_features : (M, 15) features of weather noise clusters
        model_type       : 'logistic' or 'gradient_boost'
        """
        X = np.vstack([clean_features, weather_features])
        y = np.concatenate([
            np.zeros(len(clean_features),   dtype=int),   # 0 = real
            np.ones( len(weather_features), dtype=int),   # 1 = weather
        ])

        print(f"\n[WeatherFilter] Training {model_type} classifier")
        print(f"  Real object samples  : {len(clean_features)}")
        print(f"  Weather noise samples: {len(weather_features)}")

        # Augment — weather clusters are rare, balance the dataset
        rng = np.random.default_rng(42)
        X_aug = [X]; y_aug = [y]
        noise_std = 0.03 * X.std(axis=0)
        for _ in range(4):
            X_aug.append(X + rng.normal(0, noise_std, X.shape))
            y_aug.append(y.copy())
        X = np.vstack(X_aug)
        y = np.concatenate(y_aug)
        print(f"  After augmentation   : {len(X)} samples")

        self.scaler = StandardScaler()
        X_s = self.scaler.fit_transform(X)

        if model_type == 'logistic':
            self.clf = LogisticRegression(
                C=1.0, max_iter=500, random_state=42,
                class_weight='balanced'
            )
        else:
            self.clf = GradientBoostingClassifier(
                n_estimators=100, max_depth=4,
                learning_rate=0.1, random_state=42
            )

        self.clf.fit(X_s, y)

        # Cross-validation (n_jobs=1 avoids joblib worker warnings)
        cv = cross_val_score(self.clf, X_s, y, cv=5, scoring='f1', n_jobs=1)
        print(f"  CV F1: {cv.mean():.3f} ± {cv.std():.3f}")

        train_acc = self.clf.score(X_s, y)
        print(f"  Train accuracy: {train_acc*100:.1f}%")

        # Feature importance
        if model_type == 'gradient_boost':
            imp   = self.clf.feature_importances_
            top3  = np.argsort(imp)[::-1][:3]
            print(f"  Top features for weather detection:")
            for idx in top3:
                print(f"    {FEATURE_NAMES[idx]:15s}: {imp[idx]*100:.1f}%")

        self._trained = True
        self._save()
        return self.clf

    # ── Rule-based filter ─────────────────────────────────────────────────────

    def _rule_based(self, features: np.ndarray) -> np.ndarray:
        """
        Physics-based rules to identify weather clusters.

        Returns boolean array: True = weather noise, False = real object.

        Rain signature:
          - Very narrow width (< 6px) AND tall (height > width × 1.5)
          - OR short duration (< 6ms) AND small area
          - OR very high vertical speed with small width

        Snow signature:
          - Small circular blob  (aspect 0.5–2.0, radius < 12px)
          - Very slow speed  (< 1.2 px/ms)
          - Low event count  (< 40 events total)

        Real object signature (must have at least one):
          - Large bounding box  (width > 10px OR height > 15px)
          - Long lifetime  (duration > 15ms)
          - High event count  (> 80 events)
        """
        N = len(features)
        is_weather = np.zeros(N, dtype=bool)

        width    = features[:, F_WIDTH]
        height   = features[:, F_HEIGHT]
        aspect   = features[:, F_ASPECT]        # width / height
        n_ev     = features[:, F_N_EVENTS]
        dur_ms   = features[:, F_DURATION]
        speed    = features[:, F_SPEED]
        speed_y  = features[:, F_SPEED_Y]
        density  = features[:, F_DENSITY]
        pol_r    = features[:, F_POL_RATIO]

        # ── Rain rules ────────────────────────────────────────────────────────
        # Rule R1: Very narrow streak — width < 6px and taller than wide
        rain_r1 = (width < 6) & (height > width * 1.5)

        # Rule R2: Tiny area + short lifetime — fleeting small cluster
        rain_r2 = (width * height < 60) & (dur_ms < 8)

        # Rule R3: Fast vertical movement + narrow — falling drop
        rain_r3 = (speed_y > 3.0) & (width < 8)

        # Rule R4: Very short duration with low event count
        rain_r4 = (dur_ms < 5) & (n_ev < 60)

        rain_mask = rain_r1 | rain_r2 | rain_r3 | rain_r4

        # ── Snow rules ────────────────────────────────────────────────────────
        # Rule S1: Small near-circular blob with very slow speed
        snow_s1 = (aspect > 0.5) & (aspect < 2.0) & \
                  (width < 14) & (height < 14) & (speed < 1.2)

        # Rule S2: Very few events + short duration + small
        snow_s2 = (n_ev < 40) & (dur_ms < 15) & (width < 12)

        # Rule S3: Low density diffuse cluster + slow
        snow_s3 = (density < 0.05) & (speed < 1.5) & (n_ev < 50)

        snow_mask = snow_s1 | snow_s2 | snow_s3

        # ── Real object guard — override weather rules ─────────────────────
        # If a cluster is large/long-lived/event-rich → keep it regardless
        real_guard = (width > 12) | (height > 18) | \
                     (n_ev > 100) | (dur_ms > 20)

        is_weather = (rain_mask | snow_mask) & ~real_guard

        if self.verbose:
            n_rain = (rain_mask & ~real_guard).sum()
            n_snow = (snow_mask & ~real_guard & ~rain_mask).sum()
            print(f"  Rule-based: {n_rain} rain + {n_snow} snow clusters flagged")

        return is_weather

    # ── ML-based filter ───────────────────────────────────────────────────────

    def _ml_filter(self, features: np.ndarray,
                   model_type: str = 'logistic') -> np.ndarray:
        """Run the trained ML classifier to detect weather clusters."""
        if not self._trained:
            if os.path.exists(self.model_path):
                self._load()
            else:
                print(f"  [WeatherFilter] No trained model found at "
                      f"{self.model_path}")
                print(f"  Auto-training on synthetic data...")
                self._auto_train(model_type)

        X_s  = self.scaler.transform(features)
        pred = self.clf.predict(X_s)
        prob = self.clf.predict_proba(X_s)[:, 1]   # P(weather)

        is_weather = pred == 1

        if self.verbose:
            print(f"  ML filter: {is_weather.sum()} weather clusters detected")
            for i, (pw, p) in enumerate(zip(is_weather, prob)):
                if pw:
                    print(f"    Track {i:2d}: P(weather)={p:.2f}")

        return is_weather

    # ── Auto-training (generates its own training data) ───────────────────────

    def _auto_train(self, model_type='logistic'):
        """
        Generate rich, balanced training data and train the weather filter.

        Improvement over v1:
          - 10 diverse clean scenes (different seeds, object counts, durations)
            so the classifier has seen many real object shapes/speeds/sizes
          - 6 weather conditions across all 10 scenes
            so the classifier has seen rain/snow at all intensities
          - Strictly separate: only pure-weather clusters as Class 1
            and only pure-object clusters as Class 0 — no mixing

        Training data breakdown:
          Class 0 (real objects)  : 10 scenes × ~10 tracks    = ~100 samples
          Class 1 (weather noise) : 10 scenes × 6 conditions  = ~180 samples
          After 5× augmentation   : ~1,400 total samples
        """
        print("  Generating DIVERSE training data for weather filter...")
        print("  (10 clean scenes + 6 weather conditions each)")

        from event_reader import EventReader
        from noise_filter import NoiseFilter
        from detector     import EventDetector
        from weather_noise import WeatherNoise

        det = EventDetector(window_ms=5, eps_px=10,
                            min_samples=5, min_track_len=4)
        nf  = NoiseFilter(mode='multi_scale', delta_t_us=8000)

        # ── Weather conditions to simulate (rain, snow) pairs ─────────────────
        WEATHER_CONDITIONS = [
            ('light',  None),
            ('medium', None),
            ('heavy',  None),
            (None,     'light'),
            (None,     'medium'),
            ('medium', 'light'),
        ]

        # ── Scene variations for clean object diversity ────────────────────────
        # (seed, n_objects, duration_ms)
        CLEAN_SCENES = [
            (42,  6, 200),   # baseline
            (1,   4, 200),   # fewer objects
            (2,   8, 200),   # more objects
            (3,   6, 150),   # shorter recording
            (4,   6, 250),   # longer recording
            (5,   5, 200),   # different random layout
            (6,   7, 200),
            (7,   4, 180),
            (8,   6, 200),
            (9,   8, 220),
        ]

        clean_feats_all   = []
        weather_feats_all = []

        for scene_idx, (seed, n_obj, dur_ms) in enumerate(CLEAN_SCENES):
            print(f"  Scene {scene_idx+1:2d}/10 "
                  f"(seed={seed}, n_obj={n_obj}, dur={dur_ms}ms) ...",
                  end=' ', flush=True)

            # Generate clean scene
            stream_c, lbl_c = EventReader.generate_synthetic(
                n_objects=n_obj, duration_ms=dur_ms, seed=seed)

            # Filter + detect clean tracks
            clean_s, keep_c = nf.filter(stream_c)
            lbl_c2 = lbl_c[keep_c]
            tracks_c, feats_c = det.detect(clean_s, lbl_c2)

            # Keep only purely real-object tracks (gt >= 0 and < 10)
            gt_c = np.array([tr.gt_class for tr in tracks_c])
            real_mask = (gt_c >= 0) & (gt_c < 10)
            if real_mask.sum() > 0:
                clean_feats_all.append(feats_c[real_mask])

            # Generate weather noise features for this scene
            scene_weather = 0
            for rain, snow in WEATHER_CONDITIONS:
                wn = WeatherNoise(rain_intensity=rain,
                                  snow_intensity=snow,
                                  seed=seed + 100)
                noisy_s, lbl_n, stats = wn.add_weather(stream_c, lbl_c)

                # Skip if weather added almost nothing
                if stats['total_weather'] < 100:
                    continue

                noisy_f, keep_n = nf.filter(noisy_s)
                lbl_n2 = lbl_n[keep_n]

                tracks_n, feats_n = det.detect(noisy_f)
                if len(tracks_n) == 0:
                    continue

                # Label each track by majority vote of its event labels
                gt_n = np.full(len(tracks_n), -1, dtype=int)
                for i, tr in enumerate(tracks_n):
                    all_lbl = []
                    for d in tr.detections:
                        ev_labels = lbl_n2[d.event_idx]
                        all_lbl.extend(ev_labels.tolist())
                    # Only count non-background labels
                    valid_lbl = [l for l in all_lbl if l >= 0]
                    if valid_lbl:
                        counts = np.bincount(valid_lbl, minlength=12)
                        gt_n[i] = int(counts.argmax())

                # Pure weather tracks: majority label is rain(10) or snow(11)
                weather_mask = gt_n >= 10
                if weather_mask.sum() > 0:
                    weather_feats_all.append(feats_n[weather_mask])
                    scene_weather += weather_mask.sum()

            print(f"{real_mask.sum()} obj tracks, "
                  f"{scene_weather} weather tracks")

        # ── Assemble final training arrays ────────────────────────────────────
        if not clean_feats_all:
            raise RuntimeError("No clean object tracks generated.")
        if not weather_feats_all:
            print("  Warning: No weather tracks found. "
                  "Falling back to rule-based.")
            self.mode = 'rule_based'
            return

        clean_feats   = np.vstack(clean_feats_all)
        weather_feats = np.vstack(weather_feats_all)

        print(f"\n  Training data summary:")
        print(f"  Real object tracks  : {len(clean_feats):4d}  "
              f"(from {len(CLEAN_SCENES)} diverse scenes)")
        print(f"  Weather noise tracks: {len(weather_feats):4d}  "
              f"(from {len(WEATHER_CONDITIONS)} conditions × "
              f"{len(CLEAN_SCENES)} scenes)")
        print(f"  Class ratio         : "
              f"{len(clean_feats)/len(weather_feats):.2f} : 1  "
              f"(real : weather)")

        # ── Augment clean features to balance ─────────────────────────────────
        rng = np.random.default_rng(42)
        if len(clean_feats) < len(weather_feats):
            idx = rng.choice(len(clean_feats),
                             len(weather_feats), replace=True)
            clean_feats = np.vstack([clean_feats, clean_feats[idx]])

        self.train(clean_feats, weather_feats, model_type=model_type)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({'clf': self.clf, 'scaler': self.scaler,
                         'mode': self.mode}, f)
        print(f"  Weather filter saved → {self.model_path}")

    def _load(self):
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
        self.clf     = data['clf']
        self.scaler  = data['scaler']
        self._trained = True
        print(f"  Weather filter loaded from {self.model_path}")


# ─── Visualise filtering results ──────────────────────────────────────────────

def visualise_filter_results(stream, all_tracks, all_features,
                              clean_tracks, weather_tracks,
                              pred_classes, confidences,
                              path='plots/weather_filter_results.png'):
    """
    4-panel plot showing the effect of the weather filter:
      A) All DBSCAN tracks (before filter)
      B) Weather clusters rejected
      C) Clean objects kept
      D) Final classified detections
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from representations import EventRepresentation

    BG    = '#0a0e1a'
    PANEL = '#111827'
    TEXT  = '#e2e8f0'
    MUTED = '#9ca3af'

    CLASS_COLOURS = {
        0:'#ef4444', 1:'#22c55e', 2:'#f59e0b',
        3:'#a855f7', 4:'#06b6d4', 5:'#f97316',
    }
    from classifier import CLASS_NAMES

    rep   = EventRepresentation(stream)
    frame, _ = rep.event_frame()

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.patch.set_facecolor(BG)

    cmap20 = plt.cm.tab20(np.linspace(0, 1, max(len(all_tracks), 1)))

    def setup_ax(ax, title):
        ax.set_facecolor(PANEL)
        ax.imshow(frame, aspect='auto', origin='upper',
                  interpolation='nearest', alpha=0.6)
        ax.set_title(title, color=TEXT, fontsize=12,
                     fontweight='bold', pad=8)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel('x pixel', color=MUTED)
        ax.set_ylabel('y pixel', color=MUTED)
        for sp in ax.spines.values():
            sp.set_color('#2d3748')

    # A — All tracks
    setup_ax(axes[0,0],
             f'A)  All DBSCAN Tracks\n({len(all_tracks)} total detected)')
    for i, tr in enumerate(all_tracks):
        col = cmap20[i % len(cmap20)]
        mid = tr.detections[len(tr.detections)//2]
        w = mid.x_max - mid.x_min; h = mid.y_max - mid.y_min
        rect = mpatches.FancyBboxPatch(
            (mid.x_min, mid.y_min), w, h,
            boxstyle='round,pad=1', lw=1.5,
            edgecolor=col, facecolor='none')
        axes[0,0].add_patch(rect)
        cxs = [d.cx for d in tr.detections[::2]]
        cys = [d.cy for d in tr.detections[::2]]
        axes[0,0].plot(cxs, cys, '-', color=col, lw=1, alpha=0.5)

    # B — Weather clusters rejected
    setup_ax(axes[0,1],
             f'B)  Weather Clusters Rejected\n'
             f'({len(weather_tracks)} clusters removed)')
    for tr in weather_tracks:
        mid = tr.detections[len(tr.detections)//2]
        w = mid.x_max - mid.x_min; h = mid.y_max - mid.y_min
        rect = mpatches.FancyBboxPatch(
            (mid.x_min, mid.y_min), w, h,
            boxstyle='round,pad=1', lw=2,
            edgecolor='#60a5fa', facecolor='#60a5fa', alpha=0.15)
        axes[0,1].add_patch(rect)
        rect2 = mpatches.FancyBboxPatch(
            (mid.x_min, mid.y_min), w, h,
            boxstyle='round,pad=1', lw=2,
            edgecolor='#60a5fa', facecolor='none',
            linestyle='--')
        axes[0,1].add_patch(rect2)
        axes[0,1].text(mid.cx, mid.cy, '✗',
                       color='#60a5fa', ha='center', va='center',
                       fontsize=14, fontweight='bold')

    # C — Clean tracks kept
    setup_ax(axes[1,0],
             f'C)  Clean Object Tracks Kept\n'
             f'({len(clean_tracks)} objects remain)')
    for tr in clean_tracks:
        mid = tr.detections[len(tr.detections)//2]
        w = mid.x_max - mid.x_min; h = mid.y_max - mid.y_min
        rect = mpatches.FancyBboxPatch(
            (mid.x_min, mid.y_min), w, h,
            boxstyle='round,pad=1', lw=2,
            edgecolor='#22c55e', facecolor='#22c55e', alpha=0.1)
        axes[1,0].add_patch(rect)
        rect2 = mpatches.FancyBboxPatch(
            (mid.x_min, mid.y_min), w, h,
            boxstyle='round,pad=1', lw=2,
            edgecolor='#22c55e', facecolor='none')
        axes[1,0].add_patch(rect2)
        cxs = [d.cx for d in tr.detections[::2]]
        cys = [d.cy for d in tr.detections[::2]]
        axes[1,0].plot(cxs, cys, '-', color='#22c55e',
                       lw=1.5, alpha=0.6)

    # D — Final classified detections
    setup_ax(axes[1,1],
             f'D)  Final Classified Detections\n'
             f'({len(clean_tracks)} objects with class labels)')
    for i, (tr, pred, conf) in enumerate(
            zip(clean_tracks, pred_classes, confidences)):
        col  = CLASS_COLOURS.get(pred, '#ffffff')
        name = CLASS_NAMES[pred] if pred < len(CLASS_NAMES) else '?'
        mid  = tr.detections[len(tr.detections)//2]
        w = mid.x_max - mid.x_min; h = mid.y_max - mid.y_min

        rect = mpatches.FancyBboxPatch(
            (mid.x_min, mid.y_min), w, h,
            boxstyle='round,pad=1', lw=2.5,
            edgecolor=col, facecolor=col, alpha=0.12)
        axes[1,1].add_patch(rect)
        rect2 = mpatches.FancyBboxPatch(
            (mid.x_min, mid.y_min), w, h,
            boxstyle='round,pad=1', lw=2.5,
            edgecolor=col, facecolor='none')
        axes[1,1].add_patch(rect2)
        axes[1,1].text(mid.x_min, mid.y_min - 3,
                       f'{name} {conf*100:.0f}%',
                       color=col, fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2',
                                 fc=PANEL, alpha=0.85, ec=col))

    patches = [mpatches.Patch(color=CLASS_COLOURS[c],
                               label=CLASS_NAMES[c])
               for c in range(6)]
    axes[1,1].legend(handles=patches, fontsize=8, facecolor=PANEL,
                     labelcolor=TEXT, edgecolor='#374151',
                     loc='lower right')

    fig.suptitle('Weather Cluster Filter — Before vs After',
                 color=TEXT, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.01, 1, 0.96])
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  Plot saved → {path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse, sys, os
    from event_reader  import EventReader
    from noise_filter  import NoiseFilter
    from detector      import EventDetector
    from classifier    import ObjectClassifier, CLASS_NAMES
    from weather_noise import WeatherNoise

    parser = argparse.ArgumentParser(
        description='Weather cluster filter demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--rain',   type=str, default='medium',
                        choices=['none','light','medium','heavy'])
    parser.add_argument('--snow',   type=str, default='light',
                        choices=['none','light','medium','heavy'])
    parser.add_argument('--mode',   type=str, default='logistic',
                        choices=WeatherClusterFilter.MODES)
    parser.add_argument('--retrain', action='store_true',
                        help='Force retrain the weather filter')
    args = parser.parse_args()

    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    rain = None if args.rain == 'none' else args.rain
    snow = None if args.snow == 'none' else args.snow

    # ── Step 1: Generate weather scene ───────────────────────────────────────
    print("=" * 55)
    print("  WEATHER CLUSTER FILTER DEMO")
    print("=" * 55)

    print("\n[1] Generating scene with weather noise...")
    stream_clean, lbl_clean = EventReader.generate_synthetic(
        n_objects=6, seed=42)
    wn = WeatherNoise(rain_intensity=rain, snow_intensity=snow)
    stream_noisy, lbl_noisy, stats = wn.add_weather(
        stream_clean, lbl_clean)

    # ── Step 2: Noise filter ──────────────────────────────────────────────────
    print("\n[2] Running BA noise filter...")
    nf = NoiseFilter(mode='multi_scale', delta_t_us=8000)
    stream_f, keep = nf.filter(stream_noisy)
    lbl_f = lbl_noisy[keep]

    # ── Step 3: DBSCAN detection ──────────────────────────────────────────────
    print("\n[3] Running DBSCAN detection...")
    det = EventDetector(window_ms=5, eps_px=10,
                        min_samples=5, min_track_len=4)
    all_tracks, all_features = det.detect(stream_f, lbl_f)

    print(f"\n  Before weather filter: {len(all_tracks)} tracks")

    # ── Step 4: Weather filter ────────────────────────────────────────────────
    print(f"\n[4] Applying weather filter (mode={args.mode})...")

    model_path = f'models/weather_filter_{args.mode}.pkl'
    wf = WeatherClusterFilter(mode=args.mode,
                               model_path=model_path,
                               verbose=True)

    if args.retrain and os.path.exists(model_path):
        os.remove(model_path)

    clean_tracks, clean_feats, weather_tracks, w_mask = \
        wf.filter(all_tracks, all_features)

    # ── Step 5: Classify clean tracks ────────────────────────────────────────
    print(f"\n[5] Classifying {len(clean_tracks)} clean tracks...")

    clf = ObjectClassifier(model_dir='models')
    try:
        clf._load_models()
    except FileNotFoundError:
        print("  No classifier found — training on synthetic data first...")
        _, lbl_syn = EventReader.generate_synthetic(n_objects=6, seed=42)
        stream_syn, keep_syn = nf.filter(
            EventReader.generate_synthetic(n_objects=6, seed=42)[0])
        tracks_syn, feats_syn = det.detect(stream_syn)
        gt_syn = np.array([tr.gt_class for tr in tracks_syn])
        clf.train(feats_syn, gt_syn, verbose=False)

    if len(clean_feats) > 0:
        pred, conf, proba = clf.predict(clean_feats, model='mlp')

        print(f"\n  Final detections after weather filter:")
        print(f"  {'Track':<6} {'Class':<14} {'Confidence'}")
        print(f"  {'-'*6} {'-'*14} {'-'*10}")
        class_counts = {}
        for i, (p, c) in enumerate(zip(pred, conf)):
            name = CLASS_NAMES[p] if p < len(CLASS_NAMES) else '?'
            print(f"  {i:<6} {name:<14} {c*100:.1f}%")
            class_counts[name] = class_counts.get(name, 0) + 1

        print(f"\n  Objects by class:")
        for name, cnt in sorted(class_counts.items()):
            print(f"    {name:<14}: {cnt}")

        # ── Step 6: Visualise ─────────────────────────────────────────────────
        print(f"\n[6] Generating visualisation...")
        visualise_filter_results(
            stream_f, all_tracks, all_features,
            clean_tracks, weather_tracks, pred, conf)

    print(f"\n{'='*55}")
    print(f"  SUMMARY")
    print(f"{'='*55}")
    print(f"  Total DBSCAN tracks   : {len(all_tracks)}")
    print(f"  Weather clusters removed: {len(weather_tracks)}")
    print(f"  Clean objects kept    : {len(clean_tracks)}")
    if len(clean_feats) > 0:
        print(f"  Mean confidence       : {conf.mean()*100:.1f}%")
    print(f"  Plot → plots/weather_filter_results.png")
