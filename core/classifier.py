"""
Module 5: classifier.py
========================
Multi-Layer Perceptron (MLP) classifier for event camera object tracks.

Takes the 15-dimensional feature vector from Module 4 (detector.py)
and predicts one of 6 object classes:
  0 Car  |  1 Pedestrian  |  2 Cyclist  |  3 Ball  |  4 Drone  |  5 Truck

Also trains a Random Forest as a baseline comparison.

Key concepts
------------
• Feature scaling  : StandardScaler  (mean=0, std=1 per feature)
• Data augmentation: Add Gaussian noise 5× to combat small dataset
• Early stopping   : Avoids overfitting on small real-world datasets
• Cross-validation : 5-fold CV for reliable accuracy estimate
• Model persistence: Models saved as .pkl for inference without retraining
"""

import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.neural_network  import MLPClassifier
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics         import (classification_report, confusion_matrix,
                                     accuracy_score, f1_score)

CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist', 'Ball', 'Drone', 'Truck']
MODEL_DIR   = 'models'

FEATURE_NAMES = [
    'Width (px)',     'Height (px)',      'Aspect ratio',
    'Total events',   'Duration (ms)',    'Speed (px/ms)',
    'Speed X',        'Speed Y',          'Direction (rad)',
    'Polarity ratio', 'Event density',    'CX normalised',
    'CY normalised',  'Straightness',     'Oscillation amp',
]


# ─────────────────────────────────────────────────────────────────────────────
class ObjectClassifier:
    """
    Trains and evaluates MLP + Random Forest classifiers on track features.

    Parameters
    ----------
    model_dir : str   directory where trained models are saved / loaded
    """

    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self.scaler = StandardScaler()
        self.mlp    = None
        self.rf     = None
        self.gb     = None
        self._trained = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, features: np.ndarray, gt_classes: np.ndarray,
              augment_factor: int = 5, test_size: float = 0.2,
              verbose: bool = True):
        """
        Train MLP, Random Forest, and Gradient Boosting classifiers.

        Parameters
        ----------
        features      : (N, 15) feature matrix from EventDetector
        gt_classes    : (N,)    ground-truth class labels (0–5)
        augment_factor: int     how many times to augment the dataset
        test_size     : float   fraction held out for test evaluation
        verbose       : bool    print training progress

        Returns
        -------
        results : dict   accuracy / F1 / CV scores for all models
        """
        # Filter out noise tracks (gt == -1)
        valid = gt_classes >= 0
        X = features[valid].copy()
        y = gt_classes[valid].copy()

        if len(X) == 0:
            raise ValueError("No valid labelled tracks to train on.")

        if verbose:
            print(f"\n[Classifier] Training on {len(X)} labelled tracks")
            print(f"  Classes present:")
            for c in np.unique(y):
                name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f'cls{c}'
                print(f"    {name:12s}: {(y==c).sum()} tracks")

        # ── Augmentation ──────────────────────────────────────────────────────
        if augment_factor > 1:
            rng = np.random.default_rng(42)
            X_aug_parts = [X]
            y_aug_parts = [y]
            noise_scale = 0.04 * X.std(axis=0)

            for _ in range(augment_factor - 1):
                noise = rng.normal(0, noise_scale, X.shape)
                X_aug_parts.append(X + noise)
                y_aug_parts.append(y.copy())

            X = np.vstack(X_aug_parts)
            y = np.concatenate(y_aug_parts)
            if verbose:
                print(f"  After {augment_factor}× augmentation: {len(X)} samples")

        # ── Scale features ────────────────────────────────────────────────────
        X_scaled = self.scaler.fit_transform(X)

        # ── Train / test split ────────────────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None
        )
        if verbose:
            print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

        results = {}

        # ── MLP Neural Network ────────────────────────────────────────────────
        if verbose:
            print(f"\n  Training MLP  (15→128→64→32→{len(np.unique(y))})...")

        self.mlp = MLPClassifier(
            hidden_layer_sizes = (128, 64, 32),
            activation         = 'relu',
            solver             = 'adam',
            alpha              = 1e-4,
            batch_size         = min(32, len(X_train)),
            learning_rate      = 'adaptive',
            learning_rate_init = 0.001,
            max_iter           = 600,
            early_stopping     = True,
            validation_fraction= 0.1,
            n_iter_no_change   = 25,
            random_state       = 42,
            verbose            = False,
        )
        self.mlp.fit(X_train, y_train)

        mlp_train = self.mlp.score(X_train, y_train)
        mlp_test  = self.mlp.score(X_test,  y_test)
        y_pred_m  = self.mlp.predict(X_test)
        mlp_f1    = f1_score(y_test, y_pred_m, average='weighted',
                              zero_division=0)

        if verbose:
            print(f"  MLP  train={mlp_train*100:.1f}%  "
                  f"test={mlp_test*100:.1f}%  F1={mlp_f1:.3f}")

        results['mlp'] = dict(train_acc=mlp_train, test_acc=mlp_test,
                               f1=mlp_f1, y_pred=y_pred_m, y_true=y_test)

        # ── Random Forest ─────────────────────────────────────────────────────
        if verbose:
            print(f"  Training Random Forest...")

        self.rf = RandomForestClassifier(
            n_estimators = 200,
            max_depth    = None,
            min_samples_split = 2,
            random_state = 42,
            n_jobs       = -1,
        )
        self.rf.fit(X_train, y_train)

        rf_train = self.rf.score(X_train, y_train)
        rf_test  = self.rf.score(X_test,  y_test)
        y_pred_r = self.rf.predict(X_test)
        rf_f1    = f1_score(y_test, y_pred_r, average='weighted',
                             zero_division=0)

        if verbose:
            print(f"  RF   train={rf_train*100:.1f}%  "
                  f"test={rf_test*100:.1f}%  F1={rf_f1:.3f}")

        results['rf'] = dict(train_acc=rf_train, test_acc=rf_test,
                              f1=rf_f1, y_pred=y_pred_r, y_true=y_test)

        # ── Cross-validation ──────────────────────────────────────────────────
        if verbose:
            print(f"  Running 5-fold cross-validation...")

        cv = StratifiedKFold(n_splits=min(5, len(np.unique(y))),
                             shuffle=True, random_state=42)
        cv_mlp = cross_val_score(self.mlp, X_scaled, y, cv=cv, scoring='accuracy')
        cv_rf  = cross_val_score(self.rf,  X_scaled, y, cv=cv, scoring='accuracy')

        results['cv_mlp'] = cv_mlp
        results['cv_rf']  = cv_rf

        if verbose:
            print(f"  CV MLP: {cv_mlp.mean()*100:.1f}% ± {cv_mlp.std()*100:.1f}%")
            print(f"  CV RF : {cv_rf.mean()*100:.1f}% ± {cv_rf.std()*100:.1f}%")

        # ── Feature importances ────────────────────────────────────────────────
        importances = self.rf.feature_importances_
        top5 = np.argsort(importances)[::-1][:5]
        if verbose:
            print(f"\n  Top 5 features (RF importances):")
            for rank, idx in enumerate(top5, 1):
                print(f"    {rank}. {FEATURE_NAMES[idx]:20s}: "
                      f"{importances[idx]*100:.1f}%")

        results['feature_importances'] = importances

        # ── Classification report ─────────────────────────────────────────────
        present_classes = sorted(set(y_test))
        present_names   = [CLASS_NAMES[c] if c < len(CLASS_NAMES)
                           else f'cls{c}' for c in present_classes]
        if verbose:
            print(f"\n  Classification Report (MLP):")
            print(classification_report(
                y_test, y_pred_m,
                labels=present_classes,
                target_names=present_names,
                digits=3, zero_division=0
            ))

        self._trained = True
        self._save_models()
        return results

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, features: np.ndarray, model: str = 'mlp'):
        """
        Predict class for each track feature vector.

        Parameters
        ----------
        features : (N, 15) feature matrix
        model    : 'mlp' or 'rf'

        Returns
        -------
        pred_classes : (N,) predicted class indices
        confidences  : (N,) probability of the predicted class
        """
        if not self._trained:
            self._load_models()

        X = self.scaler.transform(features)
        clf = self.mlp if model == 'mlp' else self.rf

        pred   = clf.predict(X)
        proba  = clf.predict_proba(X)
        conf   = proba.max(axis=1)

        return pred, conf, proba

    def predict_single(self, feature_vec: np.ndarray, model: str = 'mlp'):
        """Convenience wrapper for a single feature vector."""
        feat = feature_vec.reshape(1, -1)
        pred, conf, proba = self.predict(feat, model)
        cls_name = CLASS_NAMES[pred[0]] if pred[0] < len(CLASS_NAMES) else '?'
        return pred[0], cls_name, conf[0], proba[0]

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_models(self):
        for name, obj in [('scaler', self.scaler),
                           ('mlp',    self.mlp),
                           ('rf',     self.rf)]:
            path = os.path.join(self.model_dir, f'{name}.pkl')
            with open(path, 'wb') as f:
                pickle.dump(obj, f)
        print(f"\n  Models saved → {self.model_dir}/")

    def _load_models(self):
        for name, attr in [('scaler', 'scaler'),
                            ('mlp',   'mlp'),
                            ('rf',    'rf')]:
            path = os.path.join(self.model_dir, f'{name}.pkl')
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Model file '{path}' not found. Run train() first.")
            with open(path, 'rb') as f:
                setattr(self, attr, pickle.load(f))
        self._trained = True
        print(f"  Models loaded from {self.model_dir}/")


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys, os
    os.makedirs('models', exist_ok=True)
    os.makedirs('data',   exist_ok=True)

    feat_path = 'data/track_features.npy'
    gt_path   = 'data/track_gt_classes.npy'

    if not os.path.exists(feat_path):
        print("No features found — running full pipeline first...")
        os.system('python3 detector.py')

    features   = np.load(feat_path)
    gt_classes = np.load(gt_path)

    print(f"Loaded features: {features.shape}")
    print(f"GT classes     : {gt_classes.shape}  unique={np.unique(gt_classes)}")

    clf     = ObjectClassifier()
    results = clf.train(features, gt_classes, augment_factor=5, verbose=True)

    # Save results for downstream
    pred, conf, proba = clf.predict(features[gt_classes >= 0])
    np.save('data/predictions.npy',   pred)
    np.save('data/confidences.npy',   conf)
    np.save('data/probabilities.npy', proba)
    print(f"\nPredictions saved → data/predictions.npy")
