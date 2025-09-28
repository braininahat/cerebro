# %% [markdown]
# # EDA

# %% [markdown]
# ## Import Libraries

import copy
from pathlib import Path
from typing import Optional

import torch
from braindecode.datasets import BaseConcatDataset
from braindecode.models import EEGNeX
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)
from eegdash.dataset import EEGChallengeDataset
from eegdash.hbn.windows import (
    add_aux_anchors,
    add_extras_columns,
    annotate_trials_with_target,
    keep_only_recordings_with,
)
from joblib import Parallel, delayed
from matplotlib.pylab import plt
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# %% [markdown]
# ## Constants

MINI_DATASET_ROOT = Path("/media/varun/braininahat/datasets/eeg2025/mini/")
EPOCH_LEN_S = 2.0
SFREQ = 100
ANCHOR = "stimulus_anchor"
SHIFT_AFTER_STIM = 0.5
WINDOW_LEN = 2.0

# Validation and test set fractions
VALID_FRAC = 0.1
TEST_FRAC = 0.1
# Random seed
SEED = 2025

SUBJECTS_TO_REMOVE = [
    "NDARWV769JM7",
    "NDARME789TD2",
    "NDARUA442ZVF",
    "NDARJP304NK1",
    "NDARTY128YLU",
    "NDARDW550GU6",
    "NDARLD243KRE",
    "NDARUJ292JXV",
    "NDARBA381JGH",
]

BATCH_SIZE = 128
NUM_WORKERS = 8

LR = 1e-3
WEIGHT_DECAY = 1e-5
N_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# %% [markdown]
# ## Load Data

dataset_ccd = EEGChallengeDataset(
    task="contrastChangeDetection",
    release="R1",
    mini=True,
    cache_dir=MINI_DATASET_ROOT,
)

# %% [markdown]
# ## Explore Data

# %%
%matplotlib qt

raw = dataset_ccd.datasets[0].raw

fig = raw.plot()

# %% [markdown]
# ## Download all

# %%


raws = Parallel(n_jobs=-1)(delayed(lambda d: d.raw)(d) for d in dataset_ccd.datasets)

# %% [markdown]
# ## Braindecode init


# %%


# %% [markdown]
# ## Epoching

# %%

# %%
transformation_offline = [
    Preprocessor(
        annotate_trials_with_target,
        target_field="rt_from_stimulus",
        epoch_length=EPOCH_LEN_S,
        require_stimulus=True,
        require_response=True,
        apply_on_array=False,
    ),
    Preprocessor(add_aux_anchors, apply_on_array=False),
]
preprocess(dataset_ccd, transformation_offline, n_jobs=1)

# %% [markdown]
# ## Filter for stimulus anchor presence

# %%
dataset = keep_only_recordings_with(ANCHOR, dataset_ccd)

# %% [markdown]
# ## Window creation

# %%
single_windows = create_windows_from_events(
    dataset,
    mapping={ANCHOR: 0},
    trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),
    trial_stop_offset_samples=int((SHIFT_AFTER_STIM + WINDOW_LEN) * SFREQ),
    window_size_samples=int(EPOCH_LEN_S * SFREQ),
    window_stride_samples=SFREQ,
    preload=True,
)

# %% [markdown]
# ## Add metadata

# %%
single_windows = add_extras_columns(
    single_windows,
    dataset,
    desc=ANCHOR,
    keys=(
        "target",
        "rt_from_stimulus",
        "rt_from_trialstart",
        "stimulus_onset",
        "response_onset",
        "correct",
        "response_type",
    ),
)

# %% [markdown]
# ## Inspect metadata

# %%
single_windows.get_metadata().head()

# %% [markdown]
# ## Target inspection


fig, ax = plt.subplots(figsize=(15, 5))
ax = single_windows.get_metadata()["target"].plot.hist(
    bins=30, ax=ax, color="lightblue"
)
ax.set_xlabel("Response Time (s)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Response Times")
plt.show()

# %% [markdown]
# ## Train Test Split (Stratified by subject)

# %%


subjects = single_windows.description["subject"].unique()
print(f"Number of subjects: {len(subjects)}")
print(f"Subjects: {subjects}")

# %% [markdown]
# ## Remove subjects

# %%

subjects = [s for s in subjects if s not in SUBJECTS_TO_REMOVE]
print(f"Number of subjects: {len(subjects)}")
print(f"Subjects: {subjects}")

# %% [markdown]
# ## Train Test Split

# %%
train_subj, valid_test_subject = train_test_split(
    subjects,
    test_size=(VALID_FRAC + TEST_FRAC),
    random_state=check_random_state(SEED),
    shuffle=True,
)

valid_subj, test_subj = train_test_split(
    valid_test_subject,
    test_size=TEST_FRAC,
    random_state=check_random_state(SEED + 1),
    shuffle=True,
)

# %% [markdown]
# ## Sanity check

# %%
assert (set(valid_subj) | set(test_subj) | set(train_subj)) == set(subjects)

# %% [markdown]
# ## Create train/valid/test splits for the windows

# %%
subject_split = single_windows.split("subject")
train_set = []
valid_set = []
test_set = []

for s in subject_split:
    if s in train_subj:
        train_set.append(subject_split[s])
    elif s in valid_subj:
        valid_set.append(subject_split[s])
    elif s in test_subj:
        test_set.append(subject_split[s])

train_set = BaseConcatDataset(train_set)
valid_set = BaseConcatDataset(valid_set)
test_set = BaseConcatDataset(test_set)

# %% [markdown]
# ## Create dataloaders

# %%


train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)
valid_loader = DataLoader(
    valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)
test_loader = DataLoader(
    test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

# %% [markdown]
# ## Build the model

# %%


model = EEGNeX(n_chans=129, n_outputs=1, n_times=2 * SFREQ, sfreq=SFREQ)

# %% [markdown]
# ## Print model

# %%
print(model)

# %% [markdown]
# ## Train the model

# %%


# %% [markdown]
# ## Define training functions


# %%
def train_one_epoch(
    dataloader: DataLoader,
    model: Module,
    loss_fn,
    optimizer,
    scheduler: Optional[LRScheduler],
    epoch: int,
    device,
    print_batch_stats: bool = True,
):
    model.train()

    total_loss = 0.0
    sum_sq_err = 0.0
    n_samples = 0

    progress_bar = tqdm(
        enumerate(dataloader), total=len(dataloader), disable=not print_batch_stats
    )

    for batch_idx, batch in progress_bar:
        # Support datasets that may return (X, y) or (X, y, ...)
        X, y = batch[0], batch[1]
        X, y = X.to(device).float(), y.to(device).float()

        optimizer.zero_grad(set_to_none=True)
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Flatten to 1D for regression metrics and accumulate squared error
        preds_flat = preds.detach().view(-1)
        y_flat = y.detach().view(-1)
        sum_sq_err += torch.sum((preds_flat - y_flat) ** 2).item()
        n_samples += y_flat.numel()

        if print_batch_stats:
            running_rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
            progress_bar.set_description(
                f"Epoch {epoch}, Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {loss.item():.6f}, RMSE: {running_rmse:.6f}"
            )

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / len(dataloader)
    rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
    return avg_loss, rmse


# %% [markdown]
# ## Define validation function


# %%
def valid_model(
    dataloader: DataLoader,
    model: Module,
    loss_fn,
    device,
    print_batch_stats: bool = True,
):
    model.eval()
    total_loss = 0.0
    sum_sq_err = 0.0
    n_batches = len(dataloader)
    n_samples = 0

    iterator = tqdm(
        enumerate(dataloader), total=n_batches, disable=not print_batch_stats
    )

    for batch_idx, batch in iterator:
        # Supports (X, y) or (X, y, ...)\n",
        X, y = batch[0], batch[1]
        X, y = X.to(device).float(), y.to(device).float()

        preds = model(X)
        batch_loss = loss_fn(preds, y).item()
        total_loss += batch_loss

        preds_flat = preds.detach().view(-1)
        y_flat = y.detach().view(-1)
        sum_sq_err += torch.sum((preds_flat - y_flat) ** 2).item()
        n_samples += y_flat.numel()

        if print_batch_stats:
            running_rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
            iterator.set_description(
                f"Val Batch {batch_idx + 1}/{n_batches}, "
                f"Loss: {batch_loss:.6f}, RMSE: {running_rmse:.6f}"
            )

    avg_loss = total_loss / n_batches if n_batches else float("nan")
    rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5

    print(f"Val RMSE: {rmse:.6f}, Val Loss: {avg_loss:.6f}\n")
    return avg_loss, rmse


# %% [markdown]
# ## Train the model

# %%

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS - 1)
loss_fn = torch.nn.MSELoss()

patience = EARLY_STOPPING_PATIENCE
min_delta = 1e-4
best_rmse = float("inf")
epochs_no_improve = 0
best_state, best_epoch = None, None

# %% [markdown]
# ## Train the model

# %%
for epoch in range(1, N_EPOCHS + 1):
    print(f"Epoch {epoch}/{N_EPOCHS}: ", end="")

    train_loss, train_rmse = train_one_epoch(
        train_loader, model, loss_fn, optimizer, scheduler, epoch, DEVICE
    )
    val_loss, val_rmse = valid_model(test_loader, model, loss_fn, DEVICE)

    print(
        f"Train RMSE: {train_rmse:.6f}, Average Train Loss: {train_loss:.6f}, Val RMSE: {val_rmse:.6f}, Average Val Loss: {val_loss:.6f}"
    )

    if val_rmse < best_rmse - min_delta:
        best_rmse = val_rmse
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(
                f"Early stopping at epoch {epoch}. Best Val RMSE: {best_rmse:.6f} (epoch {best_epoch})"
            )
            break

# %% [markdown]
# ## Save the model

# %%
from pathlib import Path
weights_dir = Path("weights")
weights_dir.mkdir(exist_ok=True)
torch.save(model.state_dict(), weights_dir / "weights_challenge_1.pt")
print("Model saved as 'weights/weights_challenge_1.pt'")
