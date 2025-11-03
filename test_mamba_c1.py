#!/usr/bin/env python
"""Quick test of Mamba Challenge 1 with correct data paths."""

import os
os.environ['WANDB_MODE'] = 'offline'

from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim

# Mamba imports
from mamba_ssm import Mamba2

# Set seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# ========== Load mini data for quick test ==========
print("Loading HBN mini data for quick test...")

# Use cerebro's data loading utilities
from cerebro.data.challenge1 import Challenge1DataModule

# Create data module with mini data only
dm = Challenge1DataModule(
    data_dir="/media/varun/OS/Users/varun/DATASETS/HBN",
    releases=["R1"],  # Just R1 mini for quick test
    batch_size=32,
    num_workers=1,
    window_len=2.0,
    shift_after_stim=0.5,
    excluded_subjects=[],
    use_mini=True,
    val_frac=0.2,
    test_frac=0.01,  # Small test split to satisfy sklearn constraint
)

dm.setup('fit')
train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# Get a sample batch to check dimensions
for batch in train_loader:
    X, y = batch
    print(f"Input shape: {X.shape}, Target shape: {y.shape}")
    print(f"Input dtype: {X.dtype}, Target dtype: {y.dtype}")
    n_channels = X.shape[1]
    n_times = X.shape[2]
    break

# ========== Simple Mamba Model ==========
class SimpleMambaModel(nn.Module):
    def __init__(self, n_channels=129, n_times=250, d_model=128):
        super().__init__()

        # Simple channel mixing
        self.channel_mixer = nn.Linear(n_channels, d_model)

        # Single Mamba2 block
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        # Output head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        # x: (batch, channels, time)
        x = x.transpose(1, 2)  # (batch, time, channels)
        x = self.channel_mixer(x)  # (batch, time, d_model)
        x = self.mamba(x)  # (batch, time, d_model)
        x = x.transpose(1, 2)  # (batch, d_model, time)
        return self.head(x)

# ========== Quick Training Test ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = SimpleMambaModel(n_channels=n_channels, n_times=n_times, d_model=128).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Train for a few batches
model.train()
print("\nTraining for 10 batches...")
for i, batch in enumerate(train_loader):
    if i >= 10:
        break

    X, y = batch
    X = X.to(device)
    y = y.to(device)  # Already has shape [batch, 1]

    y_pred = model(X)
    loss = F.mse_loss(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Batch {i+1}: Loss = {loss.item():.4f}")

# Quick validation
model.eval()
val_losses = []
predictions = []
targets = []

print("\nValidating...")
with torch.no_grad():
    for i, batch in enumerate(val_loader):
        if i >= 5:  # Just a few batches
            break

        X, y = batch
        X = X.to(device)
        y = y.to(device)  # Already has shape [batch, 1]

        y_pred = model(X)
        loss = F.mse_loss(y_pred, y)

        val_losses.append(loss.item())
        predictions.extend(y_pred.cpu().numpy())
        targets.extend(y.cpu().numpy())

# Calculate NRMSE
predictions = np.array(predictions).flatten()
targets = np.array(targets).flatten()
mse = np.mean((targets - predictions) ** 2)
rmse = np.sqrt(mse)
y_range = targets.max() - targets.min()
nrmse = rmse / y_range if y_range > 0 else rmse

print(f"\nQuick Test Results:")
print(f"Val Loss: {np.mean(val_losses):.4f}")
print(f"NRMSE: {nrmse:.4f}")
print("\n✅ Mamba model is working! Ready for full training.")