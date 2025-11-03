#!/usr/bin/env python3
"""
Demonstration: Manual patches vs LaBraM encoder's learned features are DIFFERENT
"""
import torch
from tokenizer_vq_snp_v2 import TokenizerVQSNP

# Create tokenizer with custom LaBraMEncoder
tokenizer = TokenizerVQSNP(
    n_chans=129,
    crop_len=200,
    patch_len=20,
    dim=256,
    num_codes=512,
    vq_type='gumbel',
)

# Create dummy input
x = torch.randn(2, 129, 200)  # (B, C, T)

print("="*70)
print("Demonstrating: Manual Patches vs Labram's Learned Patches")
print("="*70)

# 1. Manual patchify (what we use for FFT targets)
xp_manual = tokenizer.patchify(x)
print(f"\n1. Manual patchify (simple view):")
print(f"   Input:  {x.shape}")
print(f"   Output: {xp_manual.shape}")
print(f"   Operation: x.view(B, C, P, L)")
print(f"   First patch, first channel, first 5 values:")
print(f"   {xp_manual[0, 0, 0, :5]}")

# 2. LaBraMEncoder's learned features (using convolutional layers)
with torch.no_grad():
    # The custom LaBraMEncoder processes patches with Conv layers
    encoder_output = tokenizer.enc(xp_manual)  # (B, C, P, dim)

    print(f"\n2. LaBraMEncoder's learned features:")
    print(f"   Input patches: {xp_manual.shape}")
    print(f"   Encoder output: {encoder_output.shape}")
    print(f"   Operation: ConvBlock layers (Conv1d + LayerNorm + GELU)")
    print(f"   First patch, first channel, first 5 learned features:")
    print(f"   {encoder_output[0, 0, 0, :5]}")

# 3. Compare: Are they the same?
print(f"\n3. Comparison:")
print(f"   Manual patch shape: {xp_manual.shape}")
print(f"   Encoder features shape: {encoder_output.shape}")
print(f"   Same content? NO!")
print(f"   - Manual patches (xp): Raw signal segments")
print(f"   - Encoder output: Learned features from ConvBlocks")

# 4. Show encoder has learnable weights
print(f"\n4. LaBraMEncoder has LEARNABLE weights:")
total_params = sum(p.numel()
                   for p in tokenizer.enc.parameters() if p.requires_grad)
print(f"   Total trainable parameters: {total_params:,}")
print(f"   These weights transform raw patches into rich features!")

# 5. Full forward pass
print(f"\n5. Full tokenizer forward pass:")
z, z_q, vq_loss, codes, xp_returned = tokenizer(x)
print(f"   Input: {x.shape}")
print(f"   Returned xp (for FFT): {xp_returned.shape}")
print(f"   Encoder output z: {z.shape}")
print(f"   Quantized z_q: {z.shape}")
print(f"   Codes: {codes.shape}")

print(f"\n6. Key Insight:")
print(f"   ✓ xp_returned == xp_manual: {torch.allclose(xp_returned, xp_manual)}")
print(f"   ✗ xp_manual ≠ Encoder features (different dimensions & semantics)")
print(f"   → They serve DIFFERENT purposes:")
print(f"     - xp_manual: Raw patches for FFT supervision (ground truth)")
print(f"     - Encoder output: Rich learned features for VQ encoding")

print("\n" + "="*70)
print("Conclusion: Manual patchify is NOT redundant!")
print("Conclusion: It provides raw signal for FFT, encoder provides learned features!")
print("="*70)
