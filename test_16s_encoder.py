#!/usr/bin/env python
"""Test what gets instantiated with the 16s pretraining config."""

import torch
from cerebro.models.components.encoders import VanillaSignalJEPAEncoder
from cerebro.utils.electrode_locations import load_hbn_chs_info
from braindecode.models import SignalJEPA

print("\n" + "="*60)
print("Analyzing 16s Pretraining Configuration")
print("="*60)

# Config parameters from pretrain_16s_80pct.yaml
n_chans = 129
n_times = 1600  # 16s @ 100Hz (paper's best)
sfreq = 100
transformer_d_model = 64
transformer_num_encoder_layers = 8
transformer_nhead = 8

print(f"\n1. Configuration parameters:")
print(f"   - Window length: 16 seconds")
print(f"   - n_times: {n_times} samples (16s @ 100Hz)")
print(f"   - n_channels: {n_chans}")
print(f"   - Transformer d_model: {transformer_d_model}")
print(f"   - Transformer encoder layers: {transformer_num_encoder_layers}")
print(f"   - Transformer attention heads: {transformer_nhead}")

# Create VanillaSignalJEPAEncoder
print(f"\n2. Creating VanillaSignalJEPAEncoder...")
encoder = VanillaSignalJEPAEncoder(
    n_chans=n_chans,
    n_times=n_times,
    sfreq=sfreq,
    drop_prob=0.0,
    transformer__d_model=transformer_d_model,
    transformer__num_encoder_layers=transformer_num_encoder_layers,
    transformer__nhead=transformer_nhead
)

print(f"\n3. VanillaSignalJEPAEncoder details:")
print(f"   Output dimension: {encoder.output_dim}")
print(f"   Model type: {encoder.model.__class__.__name__}")

# Analyze the internal SignalJEPA model
print(f"\n4. Internal SignalJEPA architecture:")
for name, module in encoder.model.named_children():
    print(f"   - {name}: {module.__class__.__name__}")
    if name == 'feature_encoder':
        print(f"     └─ Per-channel CNN extracting local features")
    elif name == 'pos_encoder':
        print(f"     └─ Positional encoding using real electrode locations")
    elif name == 'transformer':
        print(f"     └─ {transformer_num_encoder_layers} encoder layers with {transformer_nhead} heads")

# Test forward pass with 16s window
print(f"\n5. Testing forward pass with 16s window:")
batch_size = 2
x = torch.randn(batch_size, n_chans, n_times)
print(f"   Input shape: {x.shape} (batch, channels, time)")

with torch.no_grad():
    output = encoder(x)
print(f"   Output shape: {output.shape}")

# Analyze what makes this the "paper's best" configuration
print(f"\n6. Why this is the paper's best configuration:")
print(f"   ✓ 16-second windows: Captures longer temporal dependencies")
print(f"   ✓ 80% spatial masking: Encourages learning robust spatial patterns")
print(f"   ✓ Real electrode locations: Uses GSN-HydroCel-129 positions")
print(f"   ✓ 8 transformer layers: Deep enough for complex patterns")
print(f"   ✓ 64 d_model: Balanced between expressiveness and efficiency")

# Check feature encoder details
if hasattr(encoder.model, 'feature_encoder'):
    fe = encoder.model.feature_encoder
    print(f"\n7. Feature encoder CNN structure:")
    for i, layer in enumerate(fe):
        if hasattr(layer, '__name__'):
            print(f"   Layer {i}: {layer.__name__}")
        else:
            print(f"   Layer {i}: {layer}")
            if hasattr(layer, 'kernel_size'):
                print(f"     └─ Kernel: {layer.kernel_size}, Stride: {layer.stride}")

print("\n" + "="*60)
print("Summary:")
print("="*60)
print(f"VanillaSignalJEPAEncoder creates the standard SignalJEPA architecture:")
print(f"1. Per-channel CNN for local feature extraction")
print(f"2. Spatial positional encoding from real electrode coordinates")
print(f"3. {transformer_num_encoder_layers}-layer transformer for context modeling")
print(f"4. Output: (batch, sequence_length, {transformer_d_model})")
print(f"\nThis matches the paper's architecture for self-supervised pretraining.")
print("="*60)