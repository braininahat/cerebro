"""Test script for Captum integration in ModelAutopsyCallback.

Tests IG and GradCAM diagnostics on a trained checkpoint.
"""

import torch
from pathlib import Path
from cerebro.models.challenge1 import Challenge1Module
from cerebro.data.challenge1 import Challenge1DataModule

# Configuration
CHECKPOINT_PATH = "outputs/challenge1/20251013_181025/checkpoints/challenge1-epoch=16-val_nrmse=1.0007.ckpt"
DATA_DIR = Path("data/full")
NUM_SAMPLES = 100

def main():
    print("=" * 60)
    print("TESTING CAPTUM INTEGRATION")
    print("=" * 60)

    # Load checkpoint
    print(f"\n📦 Loading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    model = Challenge1Module.load_from_checkpoint(CHECKPOINT_PATH)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"✓ Model loaded on {device}")

    # Setup datamodule
    print("\n📊 Setting up datamodule...")
    datamodule = Challenge1DataModule(
        data_dir=DATA_DIR,
        releases=["R1", "R2", "R3", "R4", "R6", "R7", "R8", "R9", "R10", "R11"],
        batch_size=32,
        num_workers=4,
        use_mini=False,
        seed=2025,
    )
    datamodule.setup(stage="fit")
    val_loader = datamodule.val_dataloader()
    print(f"✓ Validation set: {len(val_loader.dataset)} samples")

    # Test 1: Integrated Gradients
    print("\n" + "=" * 60)
    print("TEST 1: INTEGRATED GRADIENTS")
    print("=" * 60)

    from cerebro.diagnostics.captum_attributions import (
        compute_integrated_gradients,
        interpret_temporal_pattern,
        interpret_spatial_pattern,
    )

    print(f"Computing IG for {NUM_SAMPLES} samples...")
    ig_results = compute_integrated_gradients(
        model.model,  # Extract PyTorch model from Lightning module
        val_loader,
        device,
        num_samples=NUM_SAMPLES,
        n_steps=50,
        baseline_type="zero",
    )

    print(f"\n📈 Results:")
    print(f"  - Attributions shape: {ig_results['attributions'].shape}")
    print(f"  - Temporal profile shape: {ig_results['temporal_profile'].shape}")
    print(f"  - Spatial profile shape: {ig_results['spatial_profile'].shape}")
    print(f"  - Peak time: {ig_results['peak_time_sec']:.2f}s post-stimulus")
    print(f"  - Peak channel: {ig_results['peak_channel_idx']}")

    print(f"\n🧠 Temporal interpretation:")
    temporal_interp = interpret_temporal_pattern(
        ig_results["peak_time_sec"], ig_results["temporal_profile"]
    )
    print(temporal_interp)

    print(f"\n🗺️  Spatial interpretation:")
    spatial_interp = interpret_spatial_pattern(
        ig_results["spatial_profile"], ig_results["peak_channel_idx"]
    )
    print(spatial_interp)

    # Test 2: Layer GradCAM
    print("\n" + "=" * 60)
    print("TEST 2: LAYER GRADCAM")
    print("=" * 60)

    from cerebro.diagnostics.captum_layers import (
        compute_layer_gradcam,
        detect_conv_layers,
        interpret_layer_hierarchy,
        summarize_layer_patterns,
    )

    # Detect conv layers
    conv_layers = detect_conv_layers(model.model)
    print(f"Detected {len(conv_layers)} convolutional layers:")
    for i, layer_name in enumerate(conv_layers[:5], 1):
        print(f"  {i}. {layer_name}")
    if len(conv_layers) > 5:
        print(f"  ... ({len(conv_layers) - 5} more)")

    print(f"\nComputing GradCAM for {NUM_SAMPLES} samples...")
    gradcam_results = compute_layer_gradcam(
        model.model,
        val_loader,
        device,
        target_layers=conv_layers,
        num_samples=NUM_SAMPLES,
    )

    print(f"\n📊 Results:")
    print(f"  - Layers analyzed: {len(gradcam_results['layer_importance'])}")
    print(f"  - Most important layer: {gradcam_results['most_important_layer']}")

    print(f"\n🏆 Layer importance ranking:")
    sorted_layers = sorted(
        gradcam_results["layer_importance"].items(),
        key=lambda x: x[1],
        reverse=True
    )
    for i, (layer_name, importance) in enumerate(sorted_layers[:5], 1):
        print(f"  {i}. {layer_name}: {importance:.6f}")

    print(f"\n🔍 Layer hierarchy interpretation:")
    hierarchy_interp = interpret_layer_hierarchy(
        gradcam_results["layer_importance"],
        gradcam_results["layer_shapes"]
    )
    print(hierarchy_interp)

    # Test 3: Visualizations
    print("\n" + "=" * 60)
    print("TEST 3: VISUALIZATIONS")
    print("=" * 60)

    from cerebro.diagnostics.visualizations import (
        plot_integrated_gradients,
        plot_layer_gradcam,
        plot_layer_temporal_profiles,
    )

    output_dir = Path("outputs/captum_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating plots...")

    # IG plot
    ig_plot_path = output_dir / "integrated_gradients.png"
    plot_integrated_gradients(ig_results, ig_plot_path)
    print(f"  ✓ {ig_plot_path}")

    # GradCAM plot
    gc_plot_path = output_dir / "layer_gradcam.png"
    plot_layer_gradcam(gradcam_results, gc_plot_path)
    print(f"  ✓ {gc_plot_path}")

    # Layer temporal profiles
    layer_patterns = summarize_layer_patterns(gradcam_results["layer_attributions"])
    ltp_plot_path = output_dir / "layer_temporal_profiles.png"
    plot_layer_temporal_profiles(layer_patterns, ltp_plot_path)
    print(f"  ✓ {ltp_plot_path}")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)
    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()
