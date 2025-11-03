#!/usr/bin/env python
"""Test neuraloperator import."""
import sys
import os

print("Python executable:", sys.executable)
print("\nPython path:")
for p in sys.path[:5]:
    print(f"  {p}")

print("\nTrying to import neuraloperator...")
try:
    import neuraloperator
    print("✓ neuraloperator imported successfully")
    print(f"  Location: {neuraloperator.__file__}")

    from neuraloperator.models import FNO1d
    print("✓ FNO1d imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")

    # Check if it's in site-packages
    site_packages = None
    for p in sys.path:
        if 'site-packages' in p:
            site_packages = p
            break

    if site_packages:
        print(f"\nChecking {site_packages}...")
        neuralop_path = os.path.join(site_packages, 'neuraloperator')
        if os.path.exists(neuralop_path):
            print(f"  neuraloperator directory exists: {neuralop_path}")
        else:
            print(f"  neuraloperator directory NOT found")