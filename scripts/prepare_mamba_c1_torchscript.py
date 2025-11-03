#!/usr/bin/env python3
"""
Convert trained Mamba Challenge 1 model to TorchScript for submission.

This script loads the best checkpoint from training and converts it to a
TorchScript file that can be used in the submission without requiring
the mamba-ssm package at inference time.
"""

from pathlib import Path
import torch
import torch.nn as nn
from mamba_ssm import Mamba2

# Model architecture (copied from notebooks/mamba_c1_full.py)

class SpatialChannelEncoder(nn.Module):
    """Encode spatial relationships between EEG channels."""

    def __init__(self, n_channels, d_model, kernel_size=3, n_filters=64):
        super().__init__()

        # Learnable channel embeddings
        self.channel_embed = nn.Embedding(n_channels, d_model)

        # Depthwise separable convolution for spatial relationships
        self.spatial_conv = nn.Conv1d(
            n_channels, n_filters,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            groups=1  # Not depthwise, allows channel mixing
        )

        # Project to model dimension
        self.channel_mixer = nn.Linear(n_filters, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Args:
            x: (batch, n_channels, n_times)
        Returns:
            (batch, n_times, d_model)
        """
        batch_size, n_channels, n_times = x.shape

        # Apply spatial convolution
        x_spatial = self.spatial_conv(x)  # (batch, n_filters, n_times)

        # Transpose for mixing
        x_spatial = x_spatial.transpose(1, 2)  # (batch, n_times, n_filters)

        # Mix into model dimension
        x_mixed = self.channel_mixer(x_spatial)  # (batch, n_times, d_model)

        # Add channel embeddings
        channel_ids = torch.arange(n_channels, device=x.device)
        channel_embeds = self.channel_embed(channel_ids).mean(dim=0)
        x_mixed = x_mixed + channel_embeds

        return self.dropout(self.norm(x_mixed))


class MambaEEGModel(nn.Module):
    """Mamba2-based model for EEG response time prediction."""

    def __init__(self, cfg):
        super().__init__()

        # Spatial encoder
        self.spatial_encoder = SpatialChannelEncoder(
            n_channels=cfg.n_channels,
            d_model=cfg.d_model,
            kernel_size=cfg.spatial_kernel_size,
            n_filters=cfg.n_spatial_filters
        )

        # Mamba2 blocks with residual connections
        self.mamba_blocks = nn.ModuleList([
            Mamba2(
                d_model=cfg.d_model,
                d_state=cfg.d_state,
                expand=cfg.expand_factor,
                d_conv=cfg.conv_size,
                headdim=32,
            )
            for _ in range(cfg.n_layers)
        ])

        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(cfg.d_model) for _ in range(cfg.n_layers)
        ])

        # Output head
        self.pooler = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.regression_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 1)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, n_channels, n_times)
        Returns:
            (batch, 1) response time predictions
        """
        # Encode spatial relationships
        x = self.spatial_encoder(x)  # (batch, n_times, d_model)

        # Pass through Mamba blocks
        for i, (mamba, norm) in enumerate(zip(self.mamba_blocks, self.layer_norms)):
            residual = x
            x = mamba(x)
            x = norm(x + residual)

        # Pool and predict
        x = x.transpose(1, 2)  # (batch, d_model, n_times)
        x = self.pooler(x)
        output = self.regression_head(x)

        return output


# Configuration (must match training config)
class Config:
    n_channels = 129
    n_times = 250  # 2.5s at 100Hz
    d_model = 256
    d_state = 32
    n_layers = 6
    expand_factor = 2
    conv_size = 4
    dropout = 0.1
    spatial_kernel_size = 3
    n_spatial_filters = 128


def main():
    print("="*60)
    print("Converting Mamba Challenge 1 model to TorchScript")
    print("="*60)

    # Paths
    checkpoint_path = Path("cache/mamba_checkpoints/mamba_c1_full_20251101_222940/best_model.pt")
    output_path = Path("startkit/mamba_c1.pt")

    print(f"\n📂 Loading checkpoint from: {checkpoint_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   🖥️  Using device: {device}")

    # Load checkpoint (weights_only=False is safe here since we created this checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"   ✅ Checkpoint loaded (epoch {checkpoint['epoch']}, val NRMSE {checkpoint['val_nrmse']:.4f})")

    # Create model
    cfg = Config()
    model = MambaEEGModel(cfg).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   📊 Model parameters: {total_params:,}")

    # Test forward pass
    print(f"\n🔍 Testing model forward pass...")
    zero_input = torch.zeros((1, 129, 250), device=device)
    with torch.inference_mode():
        output = model(zero_input)
    print(f"   ✅ Forward pass successful! Output shape: {output.shape}")

    # Convert to TorchScript
    print(f"\n⚙️  Tracing model to TorchScript...")
    try:
        scripted = torch.jit.trace(model, zero_input)
        scripted = torch.jit.optimize_for_inference(scripted)
        print(f"   ✅ TorchScript conversion successful!")
    except Exception as e:
        print(f"   ❌ TorchScript tracing failed: {e}")
        raise

    # Test TorchScript model
    print(f"\n🧪 Testing TorchScript model...")
    with torch.inference_mode():
        scripted_output = scripted(zero_input)
    print(f"   ✅ TorchScript forward pass successful! Output shape: {scripted_output.shape}")

    # Verify outputs match
    diff = (output - scripted_output).abs().max().item()
    print(f"   📏 Max difference between original and TorchScript: {diff:.2e}")
    if diff > 1e-5:
        print(f"   ⚠️  Warning: Large difference detected!")

    # Save TorchScript model
    print(f"\n💾 Saving TorchScript model to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output_path))

    # Verify saved model can be loaded
    print(f"\n✅ Verifying saved model...")
    loaded = torch.jit.load(str(output_path))
    with torch.inference_mode():
        loaded_output = loaded(zero_input)
    print(f"   ✅ Loaded model works! Output shape: {loaded_output.shape}")

    # File size
    file_size = output_path.stat().st_size / (1024 * 1024)  # MB
    print(f"   📦 TorchScript file size: {file_size:.2f} MB")

    print(f"\n{'='*60}")
    print(f"✅ Conversion complete!")
    print(f"{'='*60}")
    print(f"📄 TorchScript model saved to: {output_path}")
    print(f"📊 Ready for submission with local_scoring.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
