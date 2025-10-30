"""Debug script to diagnose electrode coordinate issue."""

import numpy as np
from cerebro.utils.electrode_locations import load_hbn_chs_info

# Load electrode locations
print("Loading electrode locations...")
chs_info = load_hbn_chs_info()

print(f"\nTotal channels: {len(chs_info)}")

# Extract all coordinates
all_coords = []
for i, ch in enumerate(chs_info):
    loc = ch['loc']
    xyz = loc[3:6]  # X, Y, Z positions
    all_coords.extend(xyz)

    if i < 5:  # Print first 5 channels
        print(f"\nChannel {i}: {ch['ch_name']}")
        print(f"  Full loc (12 elements): {loc}")
        print(f"  XYZ coordinates: {xyz}")
        print(f"  Magnitude: {max(abs(x) for x in xyz):.6f}")

# Global statistics
all_coords = np.array(all_coords)
print(f"\n{'='*60}")
print("GLOBAL COORDINATE STATISTICS:")
print(f"  Min value: {all_coords.min():.6f}")
print(f"  Max value: {all_coords.max():.6f}")
print(f"  Mean: {all_coords.mean():.6f}")
print(f"  Std: {all_coords.std():.6f}")

# Check for the specific issue SignalJEPA has
per_dimension_coords = []
for ch in chs_info:
    xyz = ch['loc'][3:6]
    per_dimension_coords.append(xyz)

per_dimension_coords = np.array(per_dimension_coords)  # Shape: (129, 3)

print(f"\nPER-DIMENSION RANGES (what SignalJEPA uses):")
for dim_idx, dim_name in enumerate(['X', 'Y', 'Z']):
    coords = per_dimension_coords[:, dim_idx]
    min_val = coords.min()
    max_val = coords.max()
    print(f"  {dim_name}: [{min_val:.6f}, {max_val:.6f}], range={max_val - min_val:.6f}")

# Compute what SignalJEPA computes
coordinate_ranges = [
    (min(coords), max(coords))
    for coords in zip(*[ch['loc'][3:6] for ch in chs_info])
]
channel_mins, channel_maxs = zip(*coordinate_ranges)
global_min = min(channel_mins)
global_max = max(channel_maxs)
max_abs_coordinate = max(abs(global_min), abs(global_max))

print(f"\nSIGNALJEPA COMPUTATION:")
print(f"  coordinate_ranges: {coordinate_ranges}")
print(f"  global_min: {global_min:.6f}")
print(f"  global_max: {global_max:.6f}")
print(f"  max_abs_coordinate: {max_abs_coordinate:.6f}")
print(f"  10 * max_abs_coordinate: {10 * max_abs_coordinate:.6f}")

if max_abs_coordinate < 1e-6:
    print(f"\n⚠️  WARNING: max_abs_coordinate is near zero! This causes divide-by-zero.")
else:
    print(f"\n✓ max_abs_coordinate is non-zero. Coordinates look valid.")
    print(f"\nDivide-by-zero warning may come from a different part of SignalJEPA.")
    print(f"Let me check the actual normalization code path...")

# Test the actual normalization that SignalJEPA does
print(f"\n{'='*60}")
print("TESTING ACTUAL NORMALIZATION (what line 1008 does):")
for dim_idx, dim_name in enumerate(['X', 'Y', 'Z']):
    coords = per_dimension_coords[:, dim_idx]
    x_min = 0  # SignalJEPA hardcodes this
    x_max = 10 * max_abs_coordinate

    print(f"\n{dim_name} dimension:")
    print(f"  x_min: {x_min}")
    print(f"  x_max: {x_max:.6f}")
    print(f"  Sample values: {coords[:3]}")

    # Try the normalization
    try:
        normalized = (coords - x_min) / (x_max - x_min)
        print(f"  ✓ Normalization successful")
        print(f"  Normalized range: [{normalized.min():.6f}, {normalized.max():.6f}]")
    except Exception as e:
        print(f"  ✗ Normalization failed: {e}")
