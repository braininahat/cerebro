#!/usr/bin/env python3
"""Test script for memmap window cache implementation."""

import logging
from pathlib import Path

from cerebro.data.jepa_pretrain import JEPAPretrainDataModule

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_memmap_cache():
    """Test memmap cache with mini dataset."""
    logger.info("=" * 80)
    logger.info("Testing Memmap Window Cache")
    logger.info("=" * 80)

    # Create data module with mini dataset
    dm = JEPAPretrainDataModule(
        data_dir=Path("/media/varun/OS/Users/varun/DATASETS/HBN"),
        releases=["R1"],  # Just R1 for quick test
        all_tasks=["restingState"],  # Just one task for quick test
        window_length=2.0,
        stride=1.0,
        crop_length=None,
        val_split=0.2,
        test_release=None,
        n_chans_select=129,
        sfreq=100,
        mini=True,  # Use mini dataset for fast testing
        batch_size=64,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=False
    )

    # Setup (builds cache if needed)
    logger.info("\n" + "="*80)
    logger.info("Setting up DataModule (will build cache if missing)...")
    logger.info("="*80 + "\n")
    dm.setup("fit")

    # Get train dataset
    train_ds = dm.train_dataset
    logger.info("\n" + "="*80)
    logger.info("Dataset Info:")
    logger.info(f"  Type: {type(train_ds).__name__}")
    logger.info(f"  Total windows: {len(train_ds)}")
    logger.info(f"  Stats: {train_ds.get_stats()}")
    logger.info("="*80 + "\n")

    # Test loading a few windows
    logger.info("Testing window loading (should be fast with memmap)...")
    import time

    indices = [0, 100, 200, 300, 400] if len(train_ds) > 400 else list(range(min(5, len(train_ds))))

    start = time.time()
    for idx in indices:
        window = train_ds[idx]
        logger.info(f"  Window {idx}: shape={window.shape}, dtype={window.dtype}")
    elapsed = time.time() - start

    logger.info(f"\n✓ Loaded {len(indices)} windows in {elapsed:.3f}s ({elapsed/len(indices)*1000:.1f}ms per window)")

    # Create dataloader and test iteration
    logger.info("\nTesting DataLoader iteration...")
    from torch.utils.data import DataLoader

    loader = DataLoader(
        train_ds,
        batch_size=64,
        num_workers=4,
        shuffle=True,
        prefetch_factor=2
    )

    start = time.time()
    batch = next(iter(loader))
    elapsed = time.time() - start

    logger.info(f"  First batch: shape={batch.shape}, dtype={batch.dtype}")
    logger.info(f"  Time to load: {elapsed:.3f}s")

    logger.info("\n" + "="*80)
    logger.info("✓ Memmap cache test PASSED!")
    logger.info("="*80)

if __name__ == "__main__":
    test_memmap_cache()
