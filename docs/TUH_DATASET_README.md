# TUH EEG Dataset Processing with HDF5

This guide explains how to process and use the Temple University Hospital (TUH) EEG Corpus (1.7TB, 69k+ recordings) with the cerebro project.

## Problem

The TUH EEG v2.0.1 dataset uses a directory structure that is incompatible with braindecode's built-in TUH loader:

```
TUH v2.0.1:  .../s001_2015/01_tcp_ar/aaaaaaaa_s001_t000.edf  ❌ (year only)
Braindecode: .../s001_2015_12_30/01_tcp_ar/aaaaaaaa_s001_t000.edf  ✓ (full date)
```

Additionally, loading 69k EDF files individually for each experiment is inefficient.

## Solution

We provide an HDF5-based solution that:

1. **Processes once**: Convert all 69k EDF files to a single compressed HDF5 file (~500GB estimated)
2. **Loads fast**: HDF5 provides efficient random access to recordings
3. **Braindecode-compatible**: Fully integrated with braindecode's preprocessing and windowing pipelines
4. **Resumable**: Can pause and resume processing (important for 69k files)
5. **Flexible**: Filter by subjects, sessions, montages before loading

## Components

- **`scripts/process_tuh_to_hdf5.py`**: Converts TUH EDF files to HDF5
- **`cerebro/data/tuh.py`**: Braindecode-compatible dataset classes
- **`scripts/test_tuh_hdf5.py`**: Test suite (validates on 10-file subset)
- **`notebooks/test.ipynb`**: Interactive examples

## Quick Start

### Step 1: Test on Small Subset

Verify everything works before processing the full dataset:

```bash
cd /projects/academic/wenyaoxu/anarghya/research/cerebro
python scripts/test_tuh_hdf5.py
```

Expected output: `✓ All tests passed!` (takes ~1 minute)

### Step 2: Process Full Dataset

**WARNING**: This will take several hours and create a ~500GB HDF5 file.

```bash
python scripts/process_tuh_to_hdf5.py \
    --tuh-dir /projects/academic/wenyaoxu/anarghya/research/eeg-data/tuh/tueg/v2.0.1 \
    --output-hdf5 /projects/academic/wenyaoxu/anarghya/research/eeg-data/tuh_eeg_processed.h5 \
    --n-jobs 16 \
    --compression gzip \
    --compression-level 4
```

**Parameters**:
- `--n-jobs`: Number of parallel workers (adjust based on CPU cores)
- `--compression`: `gzip` (best compression), `lzf` (faster), `none` (no compression)
- `--compression-level`: 0-9 for gzip (4 is balanced, 9 is slowest/smallest)
- `--no-resume`: Start from scratch (default: resumes from last processed file)

**Estimated time**: ~4-6 hours on 16 cores

**Monitoring progress**:
```bash
# Check HDF5 file size during processing
watch -n 60 'ls -lh /path/to/tuh_eeg_processed.h5'

# Check processing logs
tail -f nohup.out  # if running with nohup
```

**Resume after interruption**:
If processing is interrupted, simply re-run the same command. It will automatically resume from the last successfully processed file.

### Step 3: Use the Dataset

```python
from cerebro.data.tuh import TUHDataset

# Load full dataset (fast - no actual data loaded yet)
dataset = TUHDataset('/path/to/tuh_eeg_processed.h5')
print(f"Total recordings: {len(dataset.datasets)}")  # 69,672

# View metadata
metadata = dataset.get_metadata()
print(metadata.columns)
# ['recording_id', 'subject_id', 'session', 'segment', 'year', 'month', 'day',
#  'montage', 'sfreq', 'n_channels', 'duration', 'ch_names', 'meas_date', 'path']

# Filter by subjects
subset = TUHDataset(hdf5_path, subjects=['aaaaaaaa', 'aaaaaaab'])

# Filter by montage
ar_dataset = TUHDataset(hdf5_path, montages=['01_tcp_ar'])

# Split by subjects (no data leakage)
train_ds, val_ds, test_ds = dataset.split_by_subjects(
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42
)

# Access raw MNE object
raw = dataset.datasets[0].raw
print(raw.info)

# Braindecode preprocessing
from braindecode.preprocessing import preprocess, Preprocessor

preprocessors = [
    Preprocessor('filter', l_freq=0.5, h_freq=40.0),
    Preprocessor('resample', sfreq=100),
]
preprocess(train_ds, preprocessors)

# Create windows
from braindecode.preprocessing import create_fixed_length_windows

windows_ds = create_fixed_length_windows(
    train_ds,
    start_offset_samples=0,
    stop_offset_samples=None,
    window_size_samples=400,  # 4 seconds at 100 Hz
    window_stride_samples=200,  # 2 seconds stride
    drop_last_window=True,
)

# PyTorch DataLoader
from torch.utils.data import DataLoader

dataloader = DataLoader(
    windows_ds,
    batch_size=32,
    shuffle=True,
    num_workers=4,
)

for X_batch, y_batch, inds in dataloader:
    print(X_batch.shape)  # (32, n_channels, 400)
    break
```

## HDF5 File Structure

```
tuh_eeg_processed.h5
├── data/
│   ├── recording_000000  # (n_channels, n_samples) float32
│   ├── recording_000001
│   └── ...
└── metadata/
    ├── recording_id      # Dataset (array)
    ├── subject_id        # Dataset (string array)
    ├── session           # Dataset (int array)
    ├── segment           # Dataset (int array)
    ├── year              # Dataset (int array)
    ├── month             # Dataset (int array)
    ├── day               # Dataset (int array)
    ├── montage           # Dataset (string array)
    ├── sfreq             # Dataset (float array)
    ├── n_channels        # Dataset (int array)
    ├── duration          # Dataset (float array)
    ├── ch_names          # Dataset (JSON string array)
    ├── meas_date         # Dataset (string array)
    └── path              # Dataset (string array)
```

## Performance Comparison

| Operation | EDF (69k files) | HDF5 (single file) |
|-----------|-----------------|-------------------|
| Initial scan | ~5-10 minutes | < 1 second |
| Load 1 recording | ~1-2 seconds | ~0.01 seconds |
| Load 1000 recordings | ~30-60 minutes | ~10-20 seconds |
| Filter by subject | Scan all files | Query metadata table |
| Disk space | ~1.7 TB | ~500 GB (gzip-4) |

## Troubleshooting

### Out of memory during processing

Reduce batch size in `process_tuh_to_hdf5.py:L170`:
```python
batch_size = 50  # Default: 100
```

### HDF5 file not closing properly

If interrupted with Ctrl+C, the HDF5 file might be locked. Restart the Python process:
```bash
pkill -f process_tuh_to_hdf5.py
```

### "Unable to open file" error

Check file permissions and disk space:
```bash
ls -lh /path/to/tuh_eeg_processed.h5
df -h /path/to/output/directory
```

### Slow loading

Use SSD storage for HDF5 file if possible. HDD is 10-100x slower for random access.

## Advanced Usage

### Custom preprocessing pipeline

```python
from cerebro.data.tuh import TUHDataset
from braindecode.preprocessing import preprocess, Preprocessor

dataset = TUHDataset(hdf5_path)

# Custom preprocessors
preprocessors = [
    Preprocessor('pick_types', eeg=True, exclude='bads'),
    Preprocessor('filter', l_freq=1.0, h_freq=40.0),
    Preprocessor('resample', sfreq=100),
    Preprocessor(lambda x: x * 1e6, apply_on_array=True),  # V to µV
]

preprocess(dataset, preprocessors, n_jobs=8)
```

### Parallel loading with DataLoader

```python
from torch.utils.data import DataLoader

# Use multiple workers for parallel loading
dataloader = DataLoader(
    windows_ds,
    batch_size=64,
    shuffle=True,
    num_workers=8,  # Parallel data loading
    pin_memory=True,  # Faster GPU transfer
    prefetch_factor=4,  # Prefetch 4 batches per worker
)
```

### Memory-mapped access (for very large datasets)

HDF5 supports memory-mapped access for minimal memory usage:

```python
import h5py

# Open in read-only mode with memory mapping
h5f = h5py.File('tuh_eeg_processed.h5', 'r', rdcc_nbytes=0)
data = h5f['data']['recording_000000'][:]  # Only loads this recording
h5f.close()
```

## Comparison with Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **HDF5** (this solution) | ✓ Fast random access<br>✓ Compression<br>✓ Single file<br>✓ Braindecode-compatible | - Requires preprocessing<br>- Large single file |
| **Raw EDFs** | ✓ Original format<br>✓ No preprocessing | - Slow loading (69k files)<br>- No compression<br>- Incompatible with braindecode |
| **Zarr** | ✓ Cloud-optimized<br>✓ Chunked storage | - More complex setup<br>- Less mature ecosystem |
| **Parquet** | ✓ Good for metadata | - Poor for large arrays<br>- No compression for signals |
| **Dask** | ✓ Lazy evaluation<br>✓ Parallel processing | - Overkill for this use case<br>- Complex API |

## Citation

If you use this TUH EEG processing pipeline, please cite:

```bibtex
@article{obeid2016temple,
  title={The Temple University Hospital EEG Data Corpus},
  author={Obeid, Iyad and Picone, Joseph},
  journal={Frontiers in Neuroscience},
  volume={10},
  pages={196},
  year={2016},
  publisher={Frontiers}
}
```

## Support

- Check `notebooks/test.ipynb` for interactive examples
- Run `python scripts/test_tuh_hdf5.py` to verify installation
- See `cerebro/data/tuh.py` for API documentation

## Next Steps

After processing TUH to HDF5:

1. **Explore dataset**: Use `notebooks/test.ipynb` to visualize recordings
2. **Preprocess**: Apply filtering, resampling, artifact rejection
3. **Window data**: Create fixed-length or event-locked windows
4. **Train models**: Use with your EEG models (EEGNeX, SignalJEPA, etc.)
5. **Benchmark**: Compare with HBN dataset for transfer learning experiments
