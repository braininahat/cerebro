# %%
import os
import pickle
from glob import glob
from pathlib import Path
import warnings

from braindecode.datasets import BaseConcatDataset
from dotenv import load_dotenv
from eegdash.dataset import EEGChallengeDataset
from eegdash.hbn.windows import (
    add_aux_anchors,
    add_extras_columns,
    annotate_trials_with_target,
    build_trial_table,
    keep_only_recordings_with,
)

# Suppress EEGChallengeDataset warning
warnings.filterwarnings("ignore", category=UserWarning, module="eegdash.dataset.dataset")

# %%
load_dotenv()

DATA_ROOT = Path(os.getenv("EEG2025_DATA_ROOT")).resolve()
MINI_DIR = Path(f"{DATA_ROOT}/mini").resolve()
FULL_DIR = Path(f"{DATA_ROOT}/full").resolve()
CONCAT_DIR = Path(f"{DATA_ROOT}/concat").resolve()
CONCAT_MINI_DIR = Path(CONCAT_DIR / "mini").resolve()
CONCAT_FULL_DIR = Path(CONCAT_DIR / "full").resolve()
RELEASES = [f"R{i}" for i in range(1, 12)]
HOLDOUT_RELEASES = ["R5"]
EXCLUDED_SUBJECTS = [
    "NDARBA381JGH",
    "NDARDW550GU6",
    "NDARJP304NK1",
    "NDARLD243KRE",
    "NDARME789TD2",
    "NDARTY128YLU",
    "NDARUA442ZVF",
    "NDARUJ292JXV",
    "NDARWV769JM7",
]
TRIAL_TASKS = [
    "contrastChangeDetection",
    "DespicableMe",
    "DiaryOfAWimpyKid",
    "FunwithFractals",
    "RestingState",
    "seqLearning6target",
    "seqLearning8target",
    "surroundSupp",
    "symbolSearch",
    "ThePresent",
]
MOVIE_TASKS = [
    "DespicableMe",
    "DiaryOfAWimpyKid",
    "FunwithFractals",
    "ThePresent",
]

# %%
mini_releases = {r: EEGChallengeDataset(release=r, cache_dir=MINI_DIR, mini=True) for r in RELEASES}

# %%
mini_releases["R1"]

# %%
len(mini_releases["R1"].datasets)
print(f"Number of subjects in R1: {len(mini_releases['R1'].subjects)}")

