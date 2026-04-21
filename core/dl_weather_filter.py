"""
dl_weather_filter.py
=====================
Deep learning weather cluster detector — ResNet-style 1D CNN on
per-cluster event sequences.

WHY NOT YOLOv4/ResNet-50 IMAGE MODELS?
---------------------------------------
YOLOv4 and ResNet-50 are designed for 2D image classification.
Event camera clusters are NOT images — they are sparse point clouds in
(x, y, t, p) space. Rendering them as images loses the temporal structure
that discriminates rain from objects.

Instead we use a ResNet-style 1D CNN that:
  - Takes a fixed-length sequence of events sorted by time
  - Each event = 4 features: (x_norm, y_norm, t_norm, p)
  - Processes temporal sequences directly (no rendering needed)
  - Learns which temporal polarity patterns = rain vs object vs snow

This is architecturally equivalent to what PointNet/PointNet++ does for
LiDAR point clouds — processing raw points, not voxelised images.

ARCHITECTURE: EventResNet
--------------------------
  Input: (batch, SEQ_LEN, 4)  — N events × [x_norm, y_norm, t_norm, p]
  → Conv1D(32) + BN + ReLU
  → ResBlock(64) × 2          — residual connections
  → ResBlock(128) × 2
  → Global Average Pooling
  → FC(64) → Dropout(0.3) → FC(3)
  Output: logits for [real_object, rain, snow]

TRAINING DATA
--------------
  Positive examples (rain/snow): cluster event sequences from weather scenes
  Negative examples (real):      cluster event sequences from clean scenes
  Each cluster → padded/truncated to SEQ_LEN=256 events

Usage:
    from dl_weather_filter import DLWeatherFilter

    dl = DLWeatherFilter(seq_len=256)
    dl.train(stream_clean, labels_clean, stream_weather, labels_weather)
    clean_tracks, removed = dl.filter(tracks, stream)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import time, os
from typing import List, Optional, Tuple

SEQ_LEN   = 256      # events per cluster (pad/truncate)
N_CLASSES = 3        # real_object=0, rain=1, snow=2
REAL_CLASSES = set(range(6))

# ── 1D ResBlock ───────────────────────────────────────────────────────────────

class ResBlock1D(nn.Module):
    """Residual block for 1D temporal sequences."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.skip  = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch),
        ) if (in_ch != out_ch or stride != 1) else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


# ── EventResNet ───────────────────────────────────────────────────────────────

class EventResNet(nn.Module):
    """
    ResNet-style 1D CNN for event cluster classification.

    Input:  (batch, seq_len, 4)  — [x_norm, y_norm, t_norm, polarity]
    Output: (batch, 3)           — logits [real, rain, snow]
    """

    def __init__(self, seq_len: int = SEQ_LEN, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, 32, 7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.layer1 = nn.Sequential(ResBlock1D(32, 64),  ResBlock1D(64, 64))
        self.layer2 = nn.Sequential(ResBlock1D(64, 128, stride=2),
                                    ResBlock1D(128, 128))
        self.layer3 = nn.Sequential(ResBlock1D(128, 256, stride=2),
                                    ResBlock1D(256, 256))
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.head   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, N_CLASSES),
        )

    def forward(self, x):
        # x: (B, L, 4) → transpose to (B, 4, L) for Conv1d
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return self.head(x)


# ── Dataset ───────────────────────────────────────────────────────────────────

class ClusterDataset(Dataset):
    """Dataset of per-cluster event sequences with weather labels."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        # sequences: (N, SEQ_LEN, 4), labels: (N,) int  0=real 1=rain 2=snow
        self.X = torch.from_numpy(sequences).float()
        self.y = torch.from_numpy(labels).long()

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ── Sequence extraction ───────────────────────────────────────────────────────

def _cluster_to_sequence(event_idx: np.ndarray, stream,
                         seq_len: int = SEQ_LEN) -> np.ndarray:
    """
    Convert a cluster's event indices to a fixed-length (seq_len, 4) array.
    Features: [x_norm, y_norm, t_norm, polarity_norm]
    """
    idx   = event_idx[event_idx < len(stream.t)]
    if len(idx) == 0:
        return np.zeros((seq_len, 4), dtype=np.float32)

    xw = stream.x[idx].astype(np.float32) / stream.W   # 0-1
    yw = stream.y[idx].astype(np.float32) / stream.H   # 0-1
    tw = stream.t[idx].astype(np.float32)
    tw = (tw - tw.min()) / max(tw.max()-tw.min(), 1.0)  # 0-1
    pw = (stream.p[idx].astype(np.float32) + 1.0) / 2.0  # 0 or 1

    order = np.argsort(stream.t[idx])   # sort by time
    seq   = np.stack([xw[order], yw[order], tw[order], pw[order]], axis=1)

    # Pad or truncate to seq_len
    if len(seq) >= seq_len:
        # Uniform subsample if too long
        step = len(seq) / seq_len
        idx_s = (np.arange(seq_len) * step).astype(int)
        return seq[idx_s].astype(np.float32)
    else:
        padded = np.zeros((seq_len, 4), dtype=np.float32)
        padded[:len(seq)] = seq
        return padded


def _build_dataset(tracks, gt_labels_per_track: np.ndarray,
                   stream, seq_len: int = SEQ_LEN,
                   augment: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (sequences, labels) arrays from a list of tracks.

    label mapping: real object=0, rain (gt>=10 even)=1, snow (gt==11)=2
    """
    seqs   = []
    labels = []

    for tr, gt in zip(tracks, gt_labels_per_track):
        all_idx = np.concatenate([d.event_idx for d in tr.detections])
        seq     = _cluster_to_sequence(all_idx, stream, seq_len)

        if gt in REAL_CLASSES:
            lbl = 0   # real object
        elif gt == 11:
            lbl = 2   # snow
        else:
            lbl = 1   # rain (label 10, or -1 mixed)

        seqs.append(seq)
        labels.append(lbl)

        # Augment: time reversal + polarity flip
        if augment and lbl != 0:   # augment weather more
            seqs.append(seq[::-1].copy())
            labels.append(lbl)

    return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.int64)


# ── Main filter class ─────────────────────────────────────────────────────────

class DLWeatherFilter:
    """
    Deep learning weather cluster filter using EventResNet.

    Classifies each track as real_object / rain / snow based on
    its raw event sequence (no hand-crafted features needed).

    Parameters
    ----------
    seq_len    : int    events per cluster (pad/truncate). Default 256.
    epochs     : int    training epochs. Default 40.
    batch_size : int    Default 32.
    lr         : float  learning rate. Default 1e-3.
    device     : str    'cpu' or 'cuda'. Auto-detected.
    model_path : str    where to save/load trained model.
    verbose    : bool
    """

    def __init__(
        self,
        seq_len    : int   = SEQ_LEN,
        epochs     : int   = 40,
        batch_size : int   = 32,
        lr         : float = 1e-3,
        device     : str   = 'auto',
        model_path : str   = 'models/dl_weather_filter.pt',
        verbose    : bool  = True,
    ):
        self.seq_len    = seq_len
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.model_path = model_path
        self.verbose    = verbose

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model    = None
        self._trained = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, tracks_clean, stream_clean,
              tracks_weather, stream_weather,
              gt_clean: np.ndarray, gt_weather: np.ndarray):
        """
        Train EventResNet on clean + weather track sequences.

        Parameters
        ----------
        tracks_clean   : list of Track from clean scene
        stream_clean   : EventStream used to detect tracks_clean
        tracks_weather : list of Track from weather scene
        stream_weather : EventStream used to detect tracks_weather
        gt_clean       : gt_class per clean track  (all should be 0-5)
        gt_weather     : gt_class per weather track (mix of 0-5, 10, 11)
        """
        t0 = time.time()
        if self.verbose:
            print(f"\n[DLWeatherFilter] Building training dataset...")

        seqs_c,  lbl_c  = _build_dataset(tracks_clean,  gt_clean,
                                          stream_clean,  self.seq_len,
                                          augment=False)
        seqs_w,  lbl_w  = _build_dataset(tracks_weather, gt_weather,
                                          stream_weather, self.seq_len,
                                          augment=True)

        X = np.concatenate([seqs_c, seqs_w], axis=0)
        y = np.concatenate([lbl_c,  lbl_w],  axis=0)

        # Shuffle
        perm = np.random.permutation(len(y))
        X, y = X[perm], y[perm]

        # 80/20 train/val split
        n_train = int(0.8 * len(y))
        X_tr, y_tr = X[:n_train], y[:n_train]
        X_va, y_va = X[n_train:], y[n_train:]

        counts = np.bincount(y_tr, minlength=N_CLASSES)
        if self.verbose:
            print(f"  Train: {n_train} samples — real={counts[0]} rain={counts[1]} snow={counts[2]}")
            print(f"  Val:   {len(y_va)} samples")
            print(f"  Device: {self.device}")

        # Class weights to handle imbalance
        total = len(y_tr)
        weights = torch.tensor(
            [total / max(counts[i], 1) for i in range(N_CLASSES)],
            dtype=torch.float32).to(self.device)
        weights = weights / weights.sum() * N_CLASSES

        ds_tr = ClusterDataset(X_tr, y_tr)
        ds_va = ClusterDataset(X_va, y_va)
        dl_tr = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=True,
                           num_workers=0, drop_last=False)
        dl_va = DataLoader(ds_va, batch_size=self.batch_size, shuffle=False,
                           num_workers=0)

        self.model = EventResNet(self.seq_len).to(self.device)
        optim   = Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        sched   = CosineAnnealingLR(optim, T_max=self.epochs, eta_min=1e-5)
        crit    = nn.CrossEntropyLoss(weight=weights)

        best_val_acc = 0.0
        best_state   = None

        for epoch in range(1, self.epochs + 1):
            # Train
            self.model.train()
            train_loss = 0.0; train_correct = 0
            for xb, yb in dl_tr:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optim.zero_grad()
                logits = self.model(xb)
                loss   = crit(logits, yb)
                loss.backward()
                optim.step()
                train_loss    += loss.item() * len(yb)
                train_correct += (logits.argmax(1) == yb).sum().item()
            sched.step()

            # Validate
            self.model.eval()
            val_correct = 0; val_total = 0
            with torch.no_grad():
                for xb, yb in dl_va:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    preds   = self.model(xb).argmax(1)
                    val_correct += (preds == yb).sum().item()
                    val_total   += len(yb)

            val_acc   = val_correct / max(val_total, 1) * 100
            train_acc = train_correct / n_train * 100

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k:v.cpu().clone()
                                for k,v in self.model.state_dict().items()}

            if self.verbose and (epoch % 5 == 0 or epoch == 1):
                print(f"  Epoch {epoch:3d}/{self.epochs}  "
                      f"loss={train_loss/n_train:.4f}  "
                      f"train={train_acc:.1f}%  val={val_acc:.1f}%  "
                      f"best_val={best_val_acc:.1f}%")

        # Restore best model
        self.model.load_state_dict(best_state)
        self.model.eval()
        self._trained = True

        # Save
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(best_state, self.model_path)
        if self.verbose:
            print(f"  Best val accuracy: {best_val_acc:.1f}%")
            print(f"  Model saved → {self.model_path}")
            print(f"  Training time: {time.time()-t0:.1f}s")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, tracks, stream) -> np.ndarray:
        """
        Return per-track class probabilities.
        Returns (N, 3) array: [P(real), P(rain), P(snow)]
        """
        if not self._trained:
            raise RuntimeError("Model not trained. Call .train() first.")

        seqs = []
        for tr in tracks:
            all_idx = np.concatenate([d.event_idx for d in tr.detections])
            seqs.append(_cluster_to_sequence(all_idx, stream, self.seq_len))

        if len(seqs) == 0:
            return np.zeros((0, N_CLASSES), dtype=np.float32)

        X   = torch.from_numpy(np.array(seqs, dtype=np.float32)).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def filter(self, tracks, stream,
               threshold: float = 0.5) -> Tuple[list, list]:
        """
        Filter weather tracks.

        Returns (clean_tracks, removed_tracks).
        A track is removed if P(rain) + P(snow) > threshold.
        """
        if not self._trained:
            raise RuntimeError("Model not trained.")

        probs = self.predict(tracks, stream)
        weather_prob = probs[:, 1] + probs[:, 2]   # P(rain) + P(snow)

        clean   = []
        removed = []
        for tr, wp in zip(tracks, weather_prob):
            if wp > threshold:
                removed.append(tr)
            else:
                clean.append(tr)

        if hasattr(self, 'verbose') and self.verbose:
            print(f"\n[DLWeatherFilter]")
            print(f"  Input:   {len(tracks)} tracks")
            print(f"  Removed: {len(removed)} (weather)")
            print(f"  Kept:    {len(clean)}")

        return clean, removed

    def load(self):
        """Load previously saved model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No model at {self.model_path}")
        state = torch.load(self.model_path, map_location=self.device)
        self.model = EventResNet(self.seq_len).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self._trained = True
        if self.verbose:
            print(f"[DLWeatherFilter] Loaded model from {self.model_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("EventResNet architecture:")
    model = EventResNet(SEQ_LEN)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    x = torch.randn(4, SEQ_LEN, 4)
    y = model(x)
    print(f"  Input shape:  {tuple(x.shape)}")
    print(f"  Output shape: {tuple(y.shape)}  (batch=4, classes=3)")
    print(f"  Classes: 0=real  1=rain  2=snow")
