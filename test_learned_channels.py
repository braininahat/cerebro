#!/usr/bin/env python
"""Quick test script to verify SignalJEPAWithLearnedChannels instantiation."""

import torch
from cerebro.models.components.encoders import SignalJEPAWithLearnedChannels

print("Testing SignalJEPAWithLearnedChannels encoder...")

# Test with HBN-like parameters (129 channels)
print("\n1. Testing with HBN parameters (129 channels)...")
encoder_hbn = SignalJEPAWithLearnedChannels(
    n_chans=129,
    n_times=200,
    sfreq=100,
    n_electrodes=128,
)
print(f"   ✓ HBN encoder created: output_dim = {encoder_hbn.output_dim}")

# Test with TUH-like parameters (21 channels)
print("\n2. Testing with TUH parameters (21 channels)...")
encoder_tuh = SignalJEPAWithLearnedChannels(
    n_chans=21,
    n_times=200,
    sfreq=100,
    n_electrodes=128,
)
print(f"   ✓ TUH encoder created: output_dim = {encoder_hbn.output_dim}")

# Test forward pass with HBN
print("\n3. Testing forward pass with HBN data (129 channels)...")
x_hbn = torch.randn(2, 129, 200)  # batch=2, channels=129, time=200
try:
    out_hbn = encoder_hbn(x_hbn)
    print(f"   ✓ HBN forward pass: input {x_hbn.shape} → output {out_hbn.shape}")
except Exception as e:
    print(f"   ✗ HBN forward pass failed: {e}")

# Test forward pass with TUH
print("\n4. Testing forward pass with TUH data (21 channels)...")
x_tuh = torch.randn(2, 21, 200)  # batch=2, channels=21, time=200
try:
    out_tuh = encoder_tuh(x_tuh)
    print(f"   ✓ TUH forward pass: input {x_tuh.shape} → output {out_tuh.shape}")
except Exception as e:
    print(f"   ✗ TUH forward pass failed: {e}")

# Test that both encoders produce same-sized outputs (128 electrode representations)
print("\n5. Verifying unified output shape...")
if out_hbn.shape[1] == out_tuh.shape[1]:
    print(f"   ✓ Both encoders produce {out_hbn.shape[1]} channel representations")
else:
    print(f"   ✗ Output mismatch: HBN={out_hbn.shape[1]}, TUH={out_tuh.shape[1]}")

print("\n✅ All tests completed!")
print(f"\nKey insight: SignalJEPAWithLearnedChannels maps variable input channels")
print(f"(21 for TUH, 129 for HBN) to a unified 128-electrode representation space.")
print(f"This enables cross-dataset pretraining without requiring channel locations!")
