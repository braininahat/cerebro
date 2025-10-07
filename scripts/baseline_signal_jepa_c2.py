#!/usr/bin/env python
"""Baseline Challenge 2 training using SignalJEPA on mini releases.

This script trains a simple regressor that feeds window embeddings from
braindecode.models.SignalJEPA into an MLP head to predict the four
psychopathology factors. It operates on the mini datasets for quick
iteration.
"""

from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
from braindecode.models import SignalJEPA
from braindecode.preprocessing import create_fixed_length_windows
from eegdash.dataset import EEGChallengeDataset

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"^\s*\[EEGChallengeDataset\] EEG 2025 Competition Data Notice",
    module="eegdash",
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

SFREQ = 100  # Mini datasets are pre-downsampled to 100 Hz
WINDOW_SECONDS = 4.0
STRIDE_SECONDS = 2.0
MAX_WINDOWS_PER_SUBJECT = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_chs_info(raw) -> List[dict]:
    chs_info = []
    n_chans = len(raw.info["chs"])
    for idx, ch in enumerate(raw.info["chs"]):
        loc = np.array(ch["loc"][:3], dtype=float)
        loc = np.nan_to_num(loc)
        if np.linalg.norm(loc) == 0.0:
            angle = 2 * np.pi * idx / max(n_chans, 1)
            loc = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=float)
        chs_info.append({"loc": loc})
    return chs_info


def extract_label(desc: dict) -> torch.Tensor | None:
    keys = ["p_factor", "attention", "internalizing", "externalizing"]
    values = []
    for key in keys:
        value = desc.get(key)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        values.append(float(value))
    return torch.tensor(values, dtype=torch.float32)


def extract_demo(desc: dict) -> torch.Tensor:
    age = float(desc.get("age", 0.0) or 0.0)
    sex = desc.get("sex", "")
    sex_bin = 1.0 if str(sex).lower() in {"m", "male"} else 0.0
    return torch.tensor([age, sex_bin], dtype=torch.float32)


def collect_windows(
    releases: Sequence[str],
    tasks: Sequence[str],
    cache_dir: Path,
    subject_filter: Iterable[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[dict]]:
    subject_set = set(subject_filter) if subject_filter is not None else None
    all_windows = []
    all_labels = []
    all_demos = []
    all_subject_ids = []
    chs_info: List[dict] | None = None

    window_size = int(WINDOW_SECONDS * SFREQ)
    stride = int(STRIDE_SECONDS * SFREQ)

    for release in releases:
        for task in tasks:
            ds = EEGChallengeDataset(
                task=task,
                release=release,
                mini=True,
                cache_dir=cache_dir,
                # ensure required metadata is loaded
                description_fields=[
                    "subject",
                    "age",
                    "sex",
                    "p_factor",
                    "attention",
                    "internalizing",
                    "externalizing",
                ],
            )
            if chs_info is None and len(ds.datasets) > 0:
                chs_info = get_chs_info(ds.datasets[0].raw)

            subject_split = ds.split("subject")
            for subject, subject_ds in subject_split.items():
                if subject_set is not None and subject not in subject_set:
                    continue

                desc = subject_ds.datasets[0].description
                label = extract_label(desc)
                if label is None:
                    continue
                demo = extract_demo(desc)

                base_concat = subject_ds  # already BaseConcatDataset
                try:
                    windows_ds = create_fixed_length_windows(
                        base_concat,
                        window_size_samples=window_size,
                        window_stride_samples=stride,
                        drop_last_window=True,
                    )
                except OSError:
                    # Skip corrupted recordings (rare in mini datasets)
                    continue
                indices = list(range(len(windows_ds)))
                if len(indices) > MAX_WINDOWS_PER_SUBJECT:
                    random.shuffle(indices)
                    indices = indices[:MAX_WINDOWS_PER_SUBJECT]

                for idx in indices:
                    sample = windows_ds[idx]
                    X = sample[0]
                    all_windows.append(torch.tensor(X, dtype=torch.float32))
                    all_labels.append(label)
                    all_demos.append(demo)
                    all_subject_ids.append(subject)

    windows_tensor = torch.stack(all_windows)
    labels_tensor = torch.stack(all_labels)
    demos_tensor = torch.stack(all_demos)
    return (
        windows_tensor,
        labels_tensor,
        demos_tensor,
        all_subject_ids,
        chs_info or [],
    )


class WindowDataset(Dataset):
    def __init__(self, windows, labels, demos, subject_ids):
        self.windows = windows
        self.labels = labels
        self.demos = demos
        self.subject_ids = subject_ids

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return (
            self.windows[idx],
            self.labels[idx],
            self.demos[idx],
            self.subject_ids[idx],
        )


class SignalJEPARegressor(nn.Module):
    def __init__(self, chs_info: List[dict], embed_dim: int = 64):
        super().__init__()
        self.encoder = SignalJEPA(
            n_chans=len(chs_info),
            n_times=int(WINDOW_SECONDS * SFREQ),
            chs_info=chs_info,
        )
        self.embed_dim = embed_dim
        self.head = nn.Sequential(
            nn.Linear(embed_dim + 2, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, X, demos):
        features = self.encoder(X)  # (B, seq_len, d_model)
        pooled = features.mean(dim=1)
        combined = torch.cat([pooled, demos], dim=-1)
        return self.head(combined)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    subject_preds = defaultdict(list)
    subject_targets = {}
    for X, y, demos, subject_ids in loader:
        X = X.to(device)
        y = y.to(device)
        demos = demos.to(device)
        preds = model(X, demos)
        for pred, target, subj in zip(preds, y, subject_ids):
            subject_preds[subj].append(pred.cpu())
            subject_targets[subj] = target.cpu()
    all_pred = []
    all_target = []
    for subj, preds in subject_preds.items():
        mean_pred = torch.stack(preds).mean(dim=0)
        all_pred.append(mean_pred)
        all_target.append(subject_targets[subj])
    pred_tensor = torch.stack(all_pred)
    target_tensor = torch.stack(all_target)
    mse = nn.functional.mse_loss(pred_tensor, target_tensor)
    rmse = torch.sqrt(mse)
    return rmse.item()


def main(args):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    cache_dir = Path(args.data_dir)
    train_releases = [
        "R1",
        "R2",
        "R3",
        "R4",
        "R6",
        "R7",
        "R8",
        "R9",
        "R10",
        "R11",
    ]
    val_releases = ["R5"]
    tasks = ["contrastChangeDetection"]

    (
        train_windows,
        train_labels,
        train_demos,
        train_subjects,
        chs_info,
    ) = collect_windows(train_releases, tasks, cache_dir)

    print(f"train_windows shape: {train_windows.shape}")
    print("train labels finite:", torch.isfinite(train_labels).all())
    if chs_info:
        locs = torch.tensor([ci["loc"] for ci in chs_info], dtype=torch.float32)
        print("chs_info finite:", torch.isfinite(locs).all())

    (
        val_windows,
        val_labels,
        val_demos,
        val_subjects,
        _,
    ) = collect_windows(val_releases, tasks, cache_dir)

    print(f"val_windows shape: {val_windows.shape}")
    print("val labels finite:", torch.isfinite(val_labels).all())

    train_dataset = WindowDataset(train_windows, train_labels, train_demos, train_subjects)
    val_dataset = WindowDataset(val_windows, val_labels, val_demos, val_subjects)

    print(f"Train windows: {len(train_dataset)}, Val windows: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = SignalJEPARegressor(chs_info).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for X, y, demos, _ in train_loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            demos = demos.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X, demos)
            if torch.isnan(preds).any():
                raise RuntimeError("NaN encountered in predictions")
            if torch.isnan(y).any():
                raise RuntimeError("NaN encountered in targets")
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
        mean_loss = total_loss / len(train_loader.dataset)
        val_rmse = evaluate(model, val_loader, DEVICE)
        if val_rmse < best_val:
            best_val = val_rmse
            torch.save(model.state_dict(), args.output)
        print(f"Epoch {epoch:02d} | train_loss={mean_loss:.4f} | val_rmse={val_rmse:.4f}")

    print(f"Best validation RMSE: {best_val:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignalJEPA baseline for Challenge 2")
    parser.add_argument("--data-dir", default=os.environ.get("EEG2025_DATA_DIR", "./data"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="weights_signal_jepa_c2.pt")
    args = parser.parse_args()
    main(args)
