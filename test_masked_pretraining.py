#!/usr/bin/env python
"""Quick test of masked autoencoder pretraining."""

import torch
from cerebro.models.components.encoders import SignalJEPAWithLearnedChannels
from cerebro.trainers.masked_pretraining import MaskedAutoencoderTrainer, SimpleDecoder

print("Testing Masked Autoencoder Pretraining...")

# Create encoder
encoder = SignalJEPAWithLearnedChannels(
    n_chans=129,
    n_times=200,
    sfreq=100,
    n_electrodes=128,
)

# Create decoder
decoder = SimpleDecoder(
    input_dim=64,  # Matches encoder output_dim
    n_chans=129,
    n_times=200,
    hidden_dim=256,
)

# Create trainer
trainer = MaskedAutoencoderTrainer(
    encoder=encoder,
    decoder=decoder,
    time_mask_ratio=0.5,
    channel_mask_ratio=0.3,
    lr=0.0001,
)

print(f"✓ Trainer created: {trainer.__class__.__name__}")
print(f"  - Encoder output_dim: {encoder.output_dim}")
print(f"  - Time mask ratio: {trainer.time_mask_ratio}")
print(f"  - Channel mask ratio: {trainer.channel_mask_ratio}")

# Test masking
x = torch.randn(2, 129, 200)  # batch=2, channels=129, time=200
masked_x, mask = trainer.apply_mask(x)

print(f"\n✓ Masking works:")
print(f"  - Input shape: {x.shape}")
print(f"  - Masked ratio: {(~mask).float().mean():.2%}")
print(f"  - Time masked per sample: ~{(~mask[0, 0, :]).sum().item()}/{mask.shape[2]}")
print(f"  - Channels masked per sample: ~{(~mask[0, :, 0]).sum().item()}/{mask.shape[1]}")

# Test forward pass
features = encoder(masked_x)
reconstructed = decoder(features)

print(f"\n✓ Forward pass works:")
print(f"  - Encoder output: {features.shape}")
print(f"  - Decoder output: {reconstructed.shape}")
print(f"  - Matches input: {reconstructed.shape == x.shape}")

# Test loss calculation
loss = torch.nn.functional.mse_loss(reconstructed[~mask], x[~mask])
print(f"\n✓ Loss calculation:")
print(f"  - MSE on masked regions: {loss.item():.4f}")

# Test a training step simulation
print(f"\n✓ Simulating training step...")
optimizer = torch.optim.AdamW(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=0.0001
)

initial_loss = loss.item()
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Forward again after one step
masked_x2, mask2 = trainer.apply_mask(x)
features2 = encoder(masked_x2)
reconstructed2 = decoder(features2)
loss2 = torch.nn.functional.mse_loss(reconstructed2[~mask2], x[~mask2])

print(f"  - Loss before step: {initial_loss:.4f}")
print(f"  - Loss after step: {loss2.item():.4f}")
print(f"  - Loss changed: {abs(initial_loss - loss2.item()) > 0.001}")

print(f"\n✅ Masked autoencoder pretraining infrastructure is ready!")
print(f"\nNext steps:")
print(f"1. Train on HBN data with masked reconstruction")
print(f"2. Monitor convergence (loss should decrease)")
print(f"3. Use pretrained encoder for downstream tasks")
print(f"4. Compare with contrastive pretraining performance")
