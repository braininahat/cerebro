#!/usr/bin/env python
"""Quick check of data shapes from cerebro DataModule."""

import os
os.environ['WANDB_MODE'] = 'offline'
import torch
from cerebro.data.challenge1 import Challenge1DataModule

# Create data module
dm = Challenge1DataModule(
    data_dir="/media/varun/OS/Users/varun/DATASETS/HBN",
    releases=["R1"],
    batch_size=32,
    num_workers=1,
    window_len=2.0,
    shift_after_stim=0.5,
    excluded_subjects=[],
    use_mini=True,
    val_frac=0.2,
    test_frac=0.01,
)

print("Setting up data...")
dm.setup('fit')

# Get one batch to check shapes
train_loader = dm.train_dataloader()
for batch in train_loader:
    X, y = batch
    print(f"\nData shapes:")
    print(f"X shape: {X.shape} (batch, channels, time)")
    print(f"y shape: {y.shape} (batch,)")
    print(f"X dtype: {X.dtype}")
    print(f"y dtype: {y.dtype}")
    print(f"X range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"y range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"\nFirst 5 target values: {y[:5].numpy()}")
    break

print("\n✅ Data check complete!")