# %%

import os
from pathlib import Path
import warnings

from dotenv import load_dotenv
from eegdash.dataset import EEGChallengeDataset
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# Suppress EEGChallengeDataset warning
warnings.filterwarnings("ignore", category=UserWarning, module="eegdash.dataset.dataset")

# %%
# load constants from .env
load_dotenv()
DATA_ROOT = Path(os.getenv("EEG2025_DATA_ROOT")).resolve()
MINI_DIR = Path(f"{DATA_ROOT}/mini").resolve()
FULL_DIR = Path(f"{DATA_ROOT}/full").resolve()
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
print(f"DATA_ROOT: {DATA_ROOT}")
print(f"MINI_DIR: {MINI_DIR}")
print(f"FULL_DIR: {FULL_DIR}")


# %%
def download_release(release, mini=True):
    dataset = EEGChallengeDataset(
        release=release, mini=mini, cache_dir=MINI_DIR if mini else FULL_DIR
    )
    return Parallel(n_jobs=-1)(delayed(lambda d: d.raw)(d) for d in dataset.datasets)


# %%
# download and materialize all releases
for release in tqdm(RELEASES):
    mini_raws = download_release(release, mini=True)
    print(f"Downloaded {len(mini_raws)} raws for {release}")
    full_raws = download_release(release, mini=False)
    print(f"Downloaded {len(full_raws)} raws for {release}")
