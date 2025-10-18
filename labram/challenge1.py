import math
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import traceback
import sys
from pytorch_lightning.loggers import WandbLogger
from typing import Tuple, Sequence
import random
import pickle
import argparse
import os
import pytorch_lightning as pl
from collections import Counter
from config_util import Config
from eegdash import EEGChallengeDataset
from braindecode.datasets import BaseConcatDataset
from tqdm import tqdm
from eegdash.hbn.windows import (
    add_extras_columns,
    annotate_trials_with_target,
    add_aux_anchors, keep_only_recordings_with)
from braindecode.preprocessing import (
    preprocess, Preprocessor,
    create_windows_from_events, create_fixed_length_windows)
from joblib import Parallel, delayed
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from modeling_finetune import EEGRegressorPL, labram_base_patch100_200_reg


class CroppedWindowDataset(Dataset):
    """Wraps a windows dataset with random or deterministic temporal cropping."""

    def __init__(self, windows_dataset: BaseConcatDataset, crop_len: int, sfreq: int = 100, mode: str = "train"):
        """
        Parameters
        ----------
        windows_dataset : BaseConcatDataset
            The input dataset of EEG windows.
        crop_len : int
            Crop length in samples (e.g., 2 * sfreq for 2-second crops).
        sfreq : int, optional
            Sampling frequency, default 100 Hz.
        mode : {"train", "val", "test"}
            Controls whether cropping is random or deterministic (center crop).
        """
        self.windows_dataset = windows_dataset
        self.crop_len = crop_len
        self.sfreq = sfreq
        self.mode = mode.lower()
        assert self.mode in {"train", "val",
                             "test"}, f"Invalid mode: {self.mode}"

    def __len__(self):
        return len(self.windows_dataset.datasets)

    def __getitem__(self, idx):
        ds = self.windows_dataset.datasets[idx]
        x, y, _ = ds[0]  # (C, T)
        C, T = x.shape

        if T > self.crop_len:
            if self.mode == "train":
                # Random crop during training
                start = np.random.randint(0, T - self.crop_len + 1)
            else:
                # Deterministic crop (center)
                start = (T - self.crop_len) // 2
            x_crop = x[:, start:start + self.crop_len]
        else:
            # Pad if too short
            x_crop = x[:, :self.crop_len]
            if x_crop.shape[1] < self.crop_len:
                pad = self.crop_len - x_crop.shape[1]
                x_crop = np.pad(x_crop, ((0, 0), (0, pad)), mode="constant")

        return torch.from_numpy(x_crop).float(), torch.tensor(y).float()


def collate_float(batch):
    """Collate function that handles (x, y) tuples"""
    x_batch = torch.stack([item[0] for item in batch], dim=0)
    y_batch = torch.stack([item[1] for item in batch], dim=0)
    return x_batch, y_batch


def load_active_task_data(use_mini: bool = True,
                          releases: Sequence[str] | None = None) -> BaseConcatDataset:
    """Load all passive task datasets from specified releases with filtering at load time

    OPTIMIZATION: Uses eegdash's built-in query parameter to filter subjects at load time,
    which is much faster than loading all data and filtering afterward.

    For even more optimization, you could use EEGDash().find() to query metadata first
    and only load recordings that meet all criteria (duration, channels, etc.).
    """
    all_datasets_list = []
    releases = list(releases) if releases is not None else list(
        Config.RELEASES)

    # Calculate total combinations for progress bar
    total_combinations = len(releases)
    task = Config.CHALLENGE_TASK
    with tqdm(total=total_combinations, desc="Loading datasets", unit="dataset") as pbar:
        for release in releases:
            try:
                pbar.set_postfix_str(f"{release}/{task}")
                dataset = EEGChallengeDataset(
                    task=task,
                    release=release,
                    cache_dir=str(Config.DATA_DIR),
                    mini=use_mini,
                    description_fields=[
                        "subject", "session", "run", "task", "age", "sex"],
                )
                all_datasets_list.append(dataset)
                pbar.set_postfix_str(
                    f"{release}/{task} ✓ {len(dataset.datasets)} recordings")
            except Exception as e:
                pbar.set_postfix_str(f"{release}/{task} ✗ Failed")
            finally:
                pbar.update(1)

    if not all_datasets_list:
        raise ValueError("No datasets were loaded successfully!")

    all_datasets = BaseConcatDataset(all_datasets_list)
    print(
        f"\n{'='*60}\nTotal recordings loaded: {len(all_datasets.datasets)}\n{'='*60}")

    # print("Preprocess dataset")

    return all_datasets


def filter_datasets(datasets: BaseConcatDataset, check_challenge2=False) -> BaseConcatDataset:
    """Apply quality filters to datasets - optimized version using metadata"""
    print(f"Filtering {len(datasets.datasets)} datasets...")

    def check_dataset(ds):
        try:
            # Subject exclusion is now handled at load time, but double-check
            subj = ds.description.get('subject', '')
            if check_challenge2 and math.isnan(ds.description["externalizing"]):
                return None
            if subj in Config.EXCLUDED_SUBJECTS:
                return None

            # Access raw metadata without triggering full S3 download
            # The raw object is lazy-loaded but basic info should be available
            raw = ds.raw
            if raw is None:
                return None

            # Quick checks on metadata (doesn't download data)
            dur = raw.times[-1]
            if dur < Config.MIN_DURATION_S:
                return None

            # Check channels
            if len(raw.ch_names) != Config.N_CHANNELS:
                return None

            return ds
        except Exception:
            # Silently skip datasets that fail - they're likely corrupted
            return None

    # Use threading for I/O-bound metadata checks with progress bar
    print("  Checking durations and channels...")
    filtered = Parallel(n_jobs=-1, backend='threading')(
        delayed(check_dataset)(ds) for ds in tqdm(
            datasets.datasets,
            desc="  Filtering datasets",
            unit="recording",
            ncols=80
        )
    )

    filtered = [ds for ds in filtered if ds is not None]
    filtered_datasets = BaseConcatDataset(filtered)
    print(
        f"\n✓ Kept {len(filtered_datasets.datasets)}/{len(datasets.datasets)} recordings")

    task_counts = Counter(
        [ds.description.task for ds in filtered_datasets.datasets])
    print("\nTask distribution:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count}")

    return filtered_datasets


def create_active_windows(datasets: BaseConcatDataset,
                          target="rt_from_stimulus", check_challenge2=False) -> BaseConcatDataset:
    single_windows = None
    if not check_challenge2:
        preprocessors = [
            Preprocessor(
                annotate_trials_with_target,
                apply_on_array=False,
                target_field=target,
                epoch_length=Config.CROP_SIZE_S,
                require_stimulus=True,
                require_response=True,
            ),
            Preprocessor(add_aux_anchors, apply_on_array=False),
        ]
        data = preprocess(datasets, preprocessors, n_jobs=-1)
        dataset_2 = keep_only_recordings_with("stimulus_anchor", data)

        # Create single-interval windows (stim-locked, long enough to include the response)
        windows = create_windows_from_events(
            dataset_2,
            mapping={"stimulus_anchor": 0},
            trial_start_offset_samples=int(
                Config.SHIFT_AFTER_STIM * Config.SFREQ),  # +0.5 s
            trial_stop_offset_samples=int(
                (Config.SHIFT_AFTER_STIM + Config.WINDOW_STRIDE_S) * Config.SFREQ
            ),  # +2.5 s
            window_size_samples=int(Config.CROP_SIZE_S * Config.SFREQ),
            window_stride_samples=Config.SFREQ,
            preload=True,
        )
        single_windows = add_extras_columns(
            windows,
            dataset_2,
            desc="stimulus_anchor",
            keys=("target", "rt_from_stimulus", "rt_from_trialstart",
                  "stimulus_onset", "response_onset", "correct", "response_type")
        )
    else:
        single_windows = create_fixed_length_windows(
            datasets,
            window_size_samples=4 * Config.SFREQ,
            window_stride_samples=2 * Config.SFREQ,
            drop_last_window=True,
        )
    return single_windows


def load_cached_active_windows(cache_file: str,
                               use_mini: bool,
                               skip_cache: bool,
                               releases: Sequence[str]) -> BaseConcatDataset:
    """Load active windows for specific releases, using a pickle cache when available."""
    releases = list(releases)
    release_label = "+".join(releases)

    if os.path.exists(cache_file) and not skip_cache:
        print(f"\n{'='*60}")
        print("Loading filtered datasets from cache:")
        print(f"  {cache_file}")
        print(f"  Releases: {release_label}")
        print(f"{'='*60}")
        try:
            with open(cache_file, 'rb') as f:
                active_windows = pickle.load(f)
            print(
                f"✓ Loaded {len(active_windows.datasets)} filtered recordings from cache")
            return active_windows
        except Exception as e:
            print(f"✗ Failed to load cache: {e}")
            print("  Reprocessing data...")

    print("Loading active task data...")
    all_datasets = load_active_task_data(use_mini=use_mini, releases=releases)

    # Filter and window the datasets
    filtered_datasets = filter_datasets(all_datasets)
    active_windows = create_active_windows(filtered_datasets)

    # Save filtered data for future runs
    print(f"\n{'='*60}")
    print("Saving filtered datasets to cache:")
    print(f"  {cache_file}")
    print(f"  Releases: {release_label}")
    print(f"{'='*60}")
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(active_windows, f)
        print("✓ Cache saved successfully")
    except Exception as e:
        print(f"✗ Failed to save cache: {e}")
        print("  Continuing without cache...")

    return active_windows


def setup_logger_and_callbacks(run_name: str, monitor: str, ckpt_dir: str, use_wandb: bool):
    """Setup logging and callbacks"""
    os.makedirs(ckpt_dir, exist_ok=True)

    if use_wandb:
        logger = WandbLogger(
            project=Config.WANDB_PROJECT,
            entity=Config.WANDB_ENTITY,
            name=run_name,
            save_dir=Config.LOG_DIR
        )
    else:
        logger = None

    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{run_name}-{{epoch:02d}}-{{{monitor}:.4f}}",
        monitor=monitor,
        mode='min',
        save_top_k=3,
        save_last=True
    )

    early_stop = EarlyStopping(
        monitor=monitor,
        patience=15,
        mode='min',
        verbose=True
    )

    callbacks = [ckpt_cb, early_stop]

    return logger, callbacks, ckpt_cb


def finetune(train_loader, val_loader, labram_ckpt: str = "best_mem.ckpt", test_loader=None):
    cb_path = os.path.join(Config.MEM_DIR, labram_ckpt)
    backbone = labram_base_patch100_200_reg(
        pretrained=True,
        init_ckpt=cb_path,
        ignore_mlp_head=True,   # drop old head if shape differs
        load_strict=False,      # be forgiving across variants
    )

    model = EEGRegressorPL(
        backbone=backbone,
        patch_size=100, lr=1e-4
    )

    logger, callbacks, ckpt_cb = setup_logger_and_callbacks(
        run_name="challenge1_labram",
        monitor="val/loss",
        ckpt_dir=Config.FINETUNE_DIR,
        use_wandb=Config.USE_WANDB
    )

    trainer = pl.Trainer(
        max_epochs=Config.MAX_EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision="32-true",
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
    )

    print("About to call trainer.fit()...")
    sys.stdout.flush()  # Force flush to ensure print appears

    try:
        trainer.fit(model, train_loader, val_loader)
        print("trainer.fit() completed successfully!")
        sys.stdout.flush()
    except KeyboardInterrupt:
        print("Training interrupted by user")
        raise
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR in trainer.fit():")
        print(f"{'='*60}")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {e}")
        print(f"{'='*60}")
        print("\nFULL STACK TRACE:")
        print(f"{'='*60}")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise

    best_ckpt = ckpt_cb.best_model_path
    print(f"✓ Tokenizer training complete: {best_ckpt}")

    test_results = None
    if test_loader is not None:
        ckpt_path = best_ckpt if best_ckpt else "last"
        print("Evaluating on held-out test set...")
        test_results = trainer.test(model, dataloaders=test_loader,
                                    ckpt_path=ckpt_path)
        if test_results:
            print("✓ Test evaluation complete.")

    return best_ckpt, test_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_mini", action="store_true", default=False,
                        help="Use mini dataset")
    parser.add_argument("--tokenizer_ckpt", type=str, default=None,
                        help="Path to tokenizer checkpoint (for dump_codes/masked stages)")
    parser.add_argument("--skip_cache", action="store_true", default=False,
                        help="Skip loading from cache and reprocess data")
    args = parser.parse_args()

    # Set seeds
    pl.seed_everything(Config.SEED)

    dataset_type = "mini" if args.use_mini else "full"
    train_cache = os.path.join(
        Config.CACHE_DIR, f"filtered_datasets_ccd_{dataset_type}.pkl")
    test_cache = os.path.join(
        Config.CACHE_DIR, f"filtered_datasets_ccd_{dataset_type}_{Config.TEST_RELEASE.lower()}.pkl")

    active_windows = load_cached_active_windows(
        cache_file=train_cache,
        use_mini=args.use_mini,
        skip_cache=args.skip_cache,
        releases=Config.RELEASES,
    )

    test_windows = load_cached_active_windows(
        cache_file=test_cache,
        use_mini=args.use_mini,
        skip_cache=args.skip_cache,
        releases=[Config.TEST_RELEASE],
    )

    # Ensure test data only comes from the designated release when metadata allows checking
    release_values = {
        ds.description.get("release")
        for ds in getattr(test_windows, "datasets", [])
        if getattr(ds, "description", None) is not None
    }
    release_values = {rel for rel in release_values if rel is not None}
    if release_values and release_values != {Config.TEST_RELEASE}:
        raise ValueError(
            f"Test set must only contain release {Config.TEST_RELEASE}, found: {sorted(release_values)}")

    meta_information = active_windows.get_metadata()
    subjects = meta_information["subject"].unique()

    train_subj, valid_subj = train_test_split(
        subjects, test_size=Config.VAL_RATIO,
        random_state=check_random_state(Config.SEED), shuffle=True)
    subject_split = active_windows.split("subject")

    train_windows = BaseConcatDataset(
        [subject_split[s] for s in train_subj])
    val_windows = BaseConcatDataset(
        [subject_split[s] for s in valid_subj])

    crop_len = int(Config.CROP_SIZE_S * Config.SFREQ)

    train_ds = CroppedWindowDataset(
        train_windows, crop_len=crop_len, sfreq=Config.SFREQ, mode="train")
    val_ds = CroppedWindowDataset(
        val_windows, crop_len=crop_len, sfreq=Config.SFREQ, mode="val")
    test_ds = CroppedWindowDataset(
        test_windows, crop_len=crop_len, sfreq=Config.SFREQ, mode="test")

    if len(test_ds) == 0:
        raise ValueError(
            f"No test windows were generated for release {Config.TEST_RELEASE}")

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE,
                              num_workers=Config.NUM_WORKERS, shuffle=True,
                              collate_fn=collate_float, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE,
                            num_workers=Config.NUM_WORKERS, shuffle=False,
                            collate_fn=collate_float, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE,
                             num_workers=Config.NUM_WORKERS, shuffle=False,
                             collate_fn=collate_float, pin_memory=True)

    best_ckpt, test_results = finetune(
        train_loader, val_loader, test_loader=test_loader)
    if test_results:
        print("Test metrics:")
        for key, value in test_results[0].items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            print(f"  {key}: {value}")

    print("All done!")


if __name__ == "__main__":
    main()
