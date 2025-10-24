# TUH EEG Tokenizer Training (EDF Backend)

This guide covers training the LaBraM tokenizer on TUH EEG data loaded directly from EDF files with preprocessing and window caching.

## EDF Backend vs HDF5 Backend

| Feature | EDF Backend (this guide) | HDF5 Backend |
|---------|-------------------------|--------------|
| **Preprocessing step** | ❌ Not needed | ✅ Required (15+ hours) |
| **Disk space** | 1.7TB (original EDFs) | 3.4TB+ (uncompressed HDF5) |
| **Flexibility** | ✅ Change preprocessing on-the-fly | ❌ Must reprocess entire dataset |
| **First load time** | ~2-4 hours (with caching) | Instant (after preprocessing) |
| **Subsequent loads** | Instant (uses cache) | Instant |
| **Recommended for** | Experimentation, prototyping | Production, final training |

**TL;DR**: Use EDF backend for flexibility and to save disk space. Use HDF5 backend for production after finalizing preprocessing.

## Quick Start

### 1. Set up environment variable

```bash
export TUH_DIR=/projects/academic/wenyaoxu/anarghya/research/eeg-data/tuh/tueg/v2.0.1
```

Or add to your `.bashrc`/`.zshrc`:
```bash
echo 'export TUH_DIR=/projects/academic/wenyaoxu/anarghya/research/eeg-data/tuh/tueg/v2.0.1' >> ~/.bashrc
```

### 2. Train the tokenizer (first run creates cache)

```bash
uv run cerebro fit --config configs/labram/codebook_tuh_edf.yaml
```

**First run**: Loads EDFs, preprocesses, creates windows, caches (~2-4 hours)
**Subsequent runs**: Loads from cache (instant)

### 3. Monitor training

- **WandB Dashboard**: Check `tuh-eeg-tokenizer` project
- **Checkpoints**: `outputs/checkpoints/tokenizer-tuh-edf/`
- **Logs**: `outputs/logs/tokenizer-tuh-edf/`
- **Cache**: `{TUH_DIR}/cache/windows_*.pkl`

## Understanding Caching

### Cache Key

The cache key is automatically generated from preprocessing parameters:

```
windows_subset_train_label_all_montage_01_tcp_ar_win20_stride10_sfreq100_bp1_maxall.pkl
         ↑            ↑         ↑                ↑      ↑        ↑       ↑   ↑
      subset      label     montage           2.0s    1.0s    100Hz   bandpass max_recs
```

### Cache Behavior

- **Same parameters** → Uses existing cache (instant load)
- **Different parameters** → Creates new cache (2-4 hour load)
- **Multiple experiments** → Multiple caches coexist

### Examples

```bash
# Run 1: Creates cache A
uv run cerebro fit --config configs/labram/codebook_tuh_edf.yaml \
    --data.init_args.sfreq 100 \
    --data.init_args.window_size_s 2.0

# Run 2: Uses cache A (instant)
uv run cerebro fit --config configs/labram/codebook_tuh_edf.yaml \
    --data.init_args.sfreq 100 \
    --data.init_args.window_size_s 2.0

# Run 3: Creates cache B (different sfreq)
uv run cerebro fit --config configs/labram/codebook_tuh_edf.yaml \
    --data.init_args.sfreq 200 \
    --data.init_args.window_size_s 2.0

# Run 4: Creates cache C (different window size)
uv run cerebro fit --config configs/labram/codebook_tuh_edf.yaml \
    --data.init_args.sfreq 100 \
    --data.init_args.window_size_s 4.0
```

### Managing Cache

```bash
# List all caches
ls -lh $TUH_DIR/cache/

# Remove specific cache
rm $TUH_DIR/cache/windows_subset_train_label_all_montage_01_tcp_ar_win20_stride10_sfreq100_bp1_maxall.pkl

# Remove all caches (force reprocessing)
rm -rf $TUH_DIR/cache/

# Check cache size
du -sh $TUH_DIR/cache/
```

## Advanced Usage

### Fast prototyping (load fewer recordings)

```bash
uv run cerebro fit --config configs/labram/codebook_tuh_edf.yaml \
    --data.init_args.max_recordings 100
```

This loads only 100 recordings instead of ~70k (much faster for testing).

### Change preprocessing on-the-fly

```bash
# Higher sampling rate
uv run cerebro fit --config configs/labram/codebook_tuh_edf.yaml \
    --data.init_args.sfreq 200

# Larger windows with more overlap
uv run cerebro fit --config configs/labram/codebook_tuh_edf.yaml \
    --data.init_args.window_size_s 4.0 \
    --data.init_args.window_stride_s 2.0

# Different bandpass filter
uv run cerebro fit --config configs/labram/codebook_tuh_edf.yaml \
    --data.init_args.l_freq 1.0 \
    --data.init_args.h_freq 40.0

# No bandpass filter
uv run cerebro fit --config configs/labram/codebook_tuh_edf.yaml \
    --data.init_args.apply_bandpass false
```

### Use all montages (variable channel counts)

```bash
uv run cerebro fit --config configs/labram/codebook_tuh_edf.yaml \
    --data.init_args.montages null
```

⚠️ **Warning**: This will mix recordings with different channel counts. Make sure your model can handle variable channels.

## Configuration Files

- **`codebook_tuh_edf.yaml`**: Main tokenizer training config with EDF backend
- **`datamodule/tuh_edf.yaml`**: Standalone TUH EDF DataModule config

## Default Preprocessing (Matching HBN Pipeline)

The default config applies **the same preprocessing as HBN** for consistency:
- **Montage filter**: `01_tcp_ar` (22 channels, consistent across recordings)
- **Duration filter**: Remove recordings < 4 seconds
- **Resampling**: 100 Hz (matches HBN dataset)
- **Bandpass filter**: 0.5-50 Hz (removes drifts and high-frequency noise)
- **Windowing**: 4s windows, 2s stride (2s overlap = 50% overlap)
- **Cropping**: Random 2s crops from 4s windows during training, center crop for val/test

**Pipeline summary**:
1. Load EDF → 2. Filter duration (≥4s) → 3. Bandpass (0.5-50 Hz) → 4. Resample (100 Hz) →
5. Create 4s windows (2s stride) → 6. Cache → 7. Apply random 2s crops in DataLoader

## Model Parameters

- **Codebook size**: 8192 vectors
- **Code dimension**: 32
- **Input size**: 200 samples (2s @ 100Hz)
- **Patch size**: 100 samples (1s @ 100Hz)
- **Encoder depth**: 24 transformer layers
- **Decoder depth**: 3 transformer layers

## TUH Directory Structure

The TUH dataset follows this structure:

```
tueg/v2.0.1/
├── edf/
│   └── {numeric_group}/        # e.g., 000, 001, 002, ...
│       └── {subject_id}/        # e.g., aaaaaaaa
│           └── {session_id}/    # e.g., s001_2015
│               └── {montage}/   # e.g., 01_tcp_ar
│                   └── {recording}.edf  # e.g., aaaaaaaa_s001_t000.edf
└── cache/              # Created automatically
    └── windows_*.pkl   # Cached preprocessed windows
```

**Example path**: `edf/000/aaaaaaaa/s001_2015/01_tcp_ar/aaaaaaaa_s001_t000.edf`

## Troubleshooting

### Q: "FileNotFoundError: TUH directory not found"
**A**: Make sure `TUH_DIR` environment variable is set and points to the correct directory containing `edf/` subdirectory.

```bash
# Check if directory exists
ls $TUH_DIR/edf/

# If not found, set correct path
export TUH_DIR=/projects/academic/wenyaoxu/anarghya/research/eeg-data/tuh/tueg/v2.0.1
```

### Q: "Out of memory during data loading"
**A**: Reduce num_workers or batch_size:

```bash
uv run cerebro fit --config configs/labram/codebook_tuh_edf.yaml \
    --data.init_args.num_workers 8 \
    --data.init_args.batch_size 256
```

### Q: "First load is very slow (~2-4 hours)"
**A**: This is expected! The first run loads all EDF files, applies preprocessing, creates windows, and caches. Subsequent runs with the same parameters will be instant.

**Speed up first load**:
- Use `max_recordings` to load fewer files for testing
- Use fewer workers: `--data.init_args.num_workers 32`
- Consider using HDF5 backend if you'll never change preprocessing

### Q: "Variable channel counts error"
**A**: You're loading multiple montages with different channel counts. Either:

1. Filter by specific montage:
   ```bash
   --data.init_args.montages "['01_tcp_ar']"
   ```

2. Or use a model that handles variable channels

### Q: "Cache is taking too much disk space"
**A**: Each cache file can be 5-20GB depending on preprocessing. Remove old caches:

```bash
# Check cache size
du -sh $TUH_DIR/cache/

# Remove all caches
rm -rf $TUH_DIR/cache/

# Or remove specific old caches
ls -lth $TUH_DIR/cache/  # List by time
rm $TUH_DIR/cache/windows_...old_cache.pkl
```

### Q: "How do I force reprocessing?"
**A**: Either delete the cache file or disable caching:

```bash
# Delete cache
rm $TUH_DIR/cache/windows_*.pkl

# Or disable caching
uv run cerebro fit --config configs/labram/codebook_tuh_edf.yaml \
    --data.init_args.use_cache false
```

## Performance Tips

1. **First experiment**: Use `max_recordings=100` to validate pipeline (~5-10 minutes)
2. **Full dataset first load**: Run overnight with caching enabled (~2-4 hours)
3. **Subsequent experiments**: Reuse cached windows (instant)
4. **Change preprocessing**: Creates new cache automatically

## Comparison: EDF vs HDF5 Space Usage

```
Original TUH EDFs:              1.7 TB
HDF5 (no compression):          3.4 TB  (2x larger!)
HDF5 (gzip level 9):           ~0.7 TB  (60% smaller)
EDF + cached windows (1 cache): 1.7 TB + ~10 GB = 1.71 TB
EDF + cached windows (5 caches): 1.7 TB + ~50 GB = 1.75 TB
```

**Recommendation**: Use EDF backend unless disk space is critically limited.
