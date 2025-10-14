# %% [markdown]
# # Data Quality Validation
#
# Run this notebook once before training to validate:
# 1. No subject leakage (train/val/test)
# 2. Label distributions (train vs val)
# 3. Input data quality (normalization, outliers, artifacts)
# 4. NaN/Inf detection
#
# **When to run**: Before starting any training experiments
# **Runtime**: ~2-5 minutes
# **Output**: Validation plots in outputs/

# %% Setup
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# Add cerebro to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cerebro.data.challenge1 import Challenge1DataModule

# Create output directory
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("DATA QUALITY VALIDATION")
print("=" * 60)

# %% Load data
print("\n📊 Loading data...")

datamodule = Challenge1DataModule(
    data_dir=REPO_ROOT / "data" / "full",
    releases=["R1", "R2", "R3", "R4", "R6", "R7", "R8", "R9", "R10", "R11"],
    batch_size=512,
    num_workers=8,
    use_mini=False,
    seed=2025,
)

datamodule.setup(stage="fit")

print(f"✓ Train windows: {len(datamodule.train_set)}")
print(f"✓ Val windows: {len(datamodule.val_set)}")
print(f"✓ Test windows: {len(datamodule.test_set) if hasattr(datamodule, 'test_set') else 'N/A'}")

# %% Subject leakage detection
print("\n" + "=" * 60)
print("1. SUBJECT LEAKAGE DETECTION")
print("=" * 60)


def extract_subjects(dataset):
    """Extract subject IDs from BaseConcatDataset.

    Uses get_metadata() to properly extract HBN subject IDs (e.g., "NDARWV769JM7")
    from the metadata DataFrame, not dataset indices.
    """
    if hasattr(dataset, 'get_metadata'):
        metadata = dataset.get_metadata()
        if 'subject' in metadata.columns:
            return metadata["subject"].unique().tolist()

    # Fallback: shouldn't reach here with braindecode datasets
    return []


train_subjects = set(extract_subjects(datamodule.train_set))
val_subjects = set(extract_subjects(datamodule.val_set))

# Test set might not exist yet (created during test time)
test_subjects = set()
if hasattr(datamodule, 'test_set') and datamodule.test_set is not None:
    test_subjects = set(extract_subjects(datamodule.test_set))

print(f"Train subjects: {len(train_subjects)}")
print(f"Val subjects: {len(val_subjects)}")
if test_subjects:
    print(f"Test subjects: {len(test_subjects)}")

# Critical: Check for overlap
train_val_overlap = train_subjects & val_subjects
print(f"\nChecking for overlap...")

if len(train_val_overlap) > 0:
    print(f"❌ LEAKAGE DETECTED: {len(train_val_overlap)} subjects in BOTH train AND val!")
    print(f"   Overlapping subjects: {sorted(list(train_val_overlap))[:10]}...")
    raise ValueError("Subject leakage detected! Fix data splits before training.")
else:
    print("✓ No train-val overlap detected")

if test_subjects:
    train_test_overlap = train_subjects & test_subjects
    val_test_overlap = val_subjects & test_subjects

    if len(train_test_overlap) > 0:
        print(f"❌ LEAKAGE: {len(train_test_overlap)} subjects in train+test!")
        raise ValueError("Train-test leakage detected!")

    if len(val_test_overlap) > 0:
        print(f"❌ LEAKAGE: {len(val_test_overlap)} subjects in val+test!")
        raise ValueError("Val-test leakage detected!")

    print("✓ No train-test or val-test overlap detected")

print("\n✅ Subject leakage check PASSED")

# %% Label distribution comparison
print("\n" + "=" * 60)
print("2. LABEL DISTRIBUTION COMPARISON")
print("=" * 60)


def extract_labels(dataset):
    """Extract RT labels from BaseConcatDataset.

    Uses get_metadata() to extract 'target' column (response time labels)
    from the metadata DataFrame.
    """
    if hasattr(dataset, 'get_metadata'):
        metadata = dataset.get_metadata()
        if 'target' in metadata.columns:
            return metadata["target"].values
    return np.array([])


train_labels = extract_labels(datamodule.train_set)
val_labels = extract_labels(datamodule.val_set)

print(f"Train labels: {len(train_labels)} samples")
print(f"Val labels: {len(val_labels)} samples")

# Kolmogorov-Smirnov test (null hypothesis: same distribution)
stat, pval = ks_2samp(train_labels, val_labels)

print(f"\nTrain label stats: mean={train_labels.mean():.3f}s, std={train_labels.std():.3f}s")
print(f"Val label stats:   mean={val_labels.mean():.3f}s, std={val_labels.std():.3f}s")
print(f"KS test statistic: {stat:.4f}")
print(f"KS test p-value: {pval:.4f}")

if pval < 0.05:
    print("⚠️  WARNING: Train and val distributions differ significantly (p < 0.05)!")
    print("   This may indicate distribution shift between splits.")
else:
    print("✓ Train and val distributions are similar (p ≥ 0.05)")

# Visualize
fig = plt.figure(figsize=(14, 5))

# Histogram comparison
ax1 = plt.subplot(1, 2, 1)
ax1.hist([train_labels, val_labels], bins=50, label=['Train', 'Val'], alpha=0.7,
         color=['#2E86AB', '#A23B72'])
ax1.axvline(train_labels.mean(), color='#2E86AB', linestyle='--', linewidth=2,
            label=f'Train mean: {train_labels.mean():.2f}s')
ax1.axvline(val_labels.mean(), color='#A23B72', linestyle='--', linewidth=2,
            label=f'Val mean: {val_labels.mean():.2f}s')
ax1.set_xlabel('Response Time (s)', fontsize=11)
ax1.set_ylabel('Count', fontsize=11)
ax1.set_title(f'Label Distribution (KS p={pval:.4f})', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Q-Q plot (check if val labels are normally distributed)
ax2 = plt.subplot(1, 2, 2)
from scipy.stats import probplot
probplot(val_labels, dist="norm", plot=ax2)
ax2.set_title('Q-Q Plot: Val Labels vs Normal Distribution', fontsize=13, fontweight='bold')
ax2.grid(alpha=0.3)

plt.tight_layout()
plot_path = OUTPUT_DIR / "data_validation_labels.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {plot_path}")

# %% Input data quality
print("\n" + "=" * 60)
print("3. INPUT DATA QUALITY")
print("=" * 60)

print("Loading a batch of training data...")
train_loader = datamodule.train_dataloader()
sample_batch = next(iter(train_loader))
X, y = sample_batch[0], sample_batch[1]  # (batch, 129, 200)

print(f"Batch shape: {X.shape}")
print(f"  - Batch size: {X.shape[0]}")
print(f"  - Channels: {X.shape[1]}")
print(f"  - Time samples: {X.shape[2]} (= 2.0s at 100Hz)")

# Per-channel statistics
channel_means = X.mean(dim=(0, 2))  # (129,) - average over batch and time
channel_stds = X.std(dim=(0, 2))    # (129,)

# Detect issues
flat_channels = (channel_stds < 0.01).sum().item()
extreme_outliers_per_ch = (X.abs() > 5).any(dim=(0, 2)).sum().item()

print(f"\nPer-Channel Analysis:")
print(f"  - Flat channels (std < 0.01): {flat_channels}")
print(f"  - Channels with extreme outliers (|value| > 5): {extreme_outliers_per_ch}")

# Check overall normalization
overall_mean = X.mean().item()
overall_std = X.std().item()

print(f"\nOverall Statistics:")
print(f"  - Mean: {overall_mean:.6f} (should be ≈ 0 if standardized)")
print(f"  - Std: {overall_std:.6f} (should be ≈ 1 if standardized)")

if abs(overall_mean) > 0.1:
    print("  ⚠️  Mean is far from 0 - data may not be properly standardized")
else:
    print("  ✓ Mean is close to 0")

if abs(overall_std - 1.0) > 0.2:
    print("  ⚠️  Std is far from 1 - data may not be properly standardized")
else:
    print("  ✓ Std is close to 1")

# Visualize per-channel statistics
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Per-channel means
ax = axes[0]
channel_means_np = channel_means.numpy()
bars = ax.bar(range(129), channel_means_np, color='#2E86AB', edgecolor='black', linewidth=0.3)
ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Expected: 0')
ax.set_xlabel('Channel Index', fontsize=11)
ax.set_ylabel('Mean', fontsize=11)
ax.set_title('Per-Channel Mean (should be ≈ 0)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

# Per-channel stds
ax = axes[1]
channel_stds_np = channel_stds.numpy()
bars = ax.bar(range(129), channel_stds_np, color='#06D6A0', edgecolor='black', linewidth=0.3)
ax.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Expected: 1')
ax.axhline(0.01, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Flat threshold: 0.01')
ax.set_xlabel('Channel Index', fontsize=11)
ax.set_ylabel('Std', fontsize=11)
ax.set_title('Per-Channel Std (should be ≈ 1)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plot_path = OUTPUT_DIR / "data_validation_channels.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {plot_path}")

# %% NaN/Inf detection
print("\n" + "=" * 60)
print("4. NaN/Inf DETECTION")
print("=" * 60)

nan_count = torch.isnan(X).sum().item()
inf_count = torch.isinf(X).sum().item()

print(f"Checking for invalid values in batch...")
print(f"  - NaN values: {nan_count}")
print(f"  - Inf values: {inf_count}")

if nan_count > 0:
    print(f"❌ CRITICAL: Found {nan_count} NaN values!")
    raise ValueError("NaN values detected in data!")

if inf_count > 0:
    print(f"❌ CRITICAL: Found {inf_count} Inf values!")
    raise ValueError("Inf values detected in data!")

print("✓ No NaN/Inf values detected")

# %% Value range check
print("\n" + "=" * 60)
print("5. VALUE RANGE CHECK")
print("=" * 60)

min_val = X.min().item()
max_val = X.max().item()
percentile_99 = torch.quantile(X.flatten(), 0.99).item()
percentile_01 = torch.quantile(X.flatten(), 0.01).item()

print(f"Value range:")
print(f"  - Min: {min_val:.3f}")
print(f"  - Max: {max_val:.3f}")
print(f"  - 1st percentile: {percentile_01:.3f}")
print(f"  - 99th percentile: {percentile_99:.3f}")

# Check for extreme values
if max_val > 10 or min_val < -10:
    print(f"⚠️  WARNING: Extreme values detected (>10 or <-10)")
    print(f"   This may indicate outliers or scaling issues")
else:
    print("✓ Value range looks reasonable for standardized data")

# Count extreme outliers
extreme_outlier_count = ((X.abs() > 5).sum().item())
extreme_outlier_pct = (extreme_outlier_count / X.numel()) * 100

print(f"\nExtreme outliers (|value| > 5σ):")
print(f"  - Count: {extreme_outlier_count} / {X.numel()}")
print(f"  - Percentage: {extreme_outlier_pct:.3f}%")

if extreme_outlier_pct > 1.0:
    print(f"  ⚠️  WARNING: >1% extreme outliers (expected <0.3% for normal dist)")
else:
    print(f"  ✓ Outlier percentage is reasonable")

# %% Summary
print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)

validation_results = {
    "subject_leakage": len(train_val_overlap) == 0,
    "label_distribution": pval >= 0.05,
    "no_nan_inf": nan_count == 0 and inf_count == 0,
    "normalization_mean": abs(overall_mean) <= 0.1,
    "normalization_std": abs(overall_std - 1.0) <= 0.2,
    "flat_channels": flat_channels == 0,
    "extreme_outliers": extreme_outlier_pct <= 1.0,
}

print("\nChecklist:")
for check, passed in validation_results.items():
    status = "✓" if passed else "⚠️ "
    print(f"  {status} {check.replace('_', ' ').title()}: {'PASS' if passed else 'WARN'}")

all_critical_passed = (
    validation_results["subject_leakage"] and
    validation_results["no_nan_inf"]
)

all_passed = all(validation_results.values())

print("\n" + "=" * 60)
if all_passed:
    print("✅ ALL VALIDATION CHECKS PASSED")
    print("Data is ready for training!")
elif all_critical_passed:
    print("⚠️  SOME WARNINGS DETECTED (but critical checks passed)")
    print("You can proceed with training, but investigate warnings.")
else:
    print("❌ CRITICAL VALIDATION FAILURES")
    print("DO NOT TRAIN until issues are resolved!")
print("=" * 60)

print(f"\n📊 Validation plots saved to: {OUTPUT_DIR}")
print(f"  - {OUTPUT_DIR}/data_validation_labels.png")
print(f"  - {OUTPUT_DIR}/data_validation_channels.png")

# %%
