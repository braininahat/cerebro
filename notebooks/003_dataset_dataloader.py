# %%
import os
from pathlib import Path

from eegdash.dataset import EEGChallengeDataset

BASE_DATA_DIR = Path(os.getenv("EEG2025_DATA_DIR", Path(__file__).resolve().parents[1] / "data")).expanduser()
BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)
FULL_DATASET_ROOT = (BASE_DATA_DIR / "full")
FULL_DATASET_ROOT.mkdir(parents=True, exist_ok=True)


# %%
dataset_ccd_releases = {
    f"R{str(release)}": EEGChallengeDataset(
        # task="contrastChangeDetection",
        cache_dir=FULL_DATASET_ROOT,
        mini=False,
        release=f"R{str(release)}",
    )
    for release in range(1, 12)
}

# %%
print(dataset_ccd_releases)


# %%
dataset_ccd_releases["R1"].description
# %%
from braindecode.datasets import BaseConcatDataset

# %%
dataset_ccd_releases["R1"].datasets

# %%
dataset_ccd_releases["R1"].datasets[0].raw

# %%
from joblib import Parallel, delayed

for release in dataset_ccd_releases:
    raws = Parallel(n_jobs=-1)(
        delayed(lambda d: d.raw)(d) for d in dataset_ccd_releases[release].datasets
    )
    print(f"Downloaded {len(raws)} raws for {release}")

# %%
dataset_ccd_releases["R1"].description["task"].unique()


# %% [markdown]
# ## Tasks

# - Passive
#     - `RestingState`
#     - `surroundSupp`
#     - MovieWatching
#         - `DespicableMe`
#         - `DiaryOfAWimpyKid`
#         - `FunwithFractals`
#         - `ThePresent`
# - Active
#     - `contrastChangeDetection`
#     - SequenceLearning
#         - `seqLearning6target`
#         - `seqLearning8target`
#     - `symbolSearch`


# %% [markdown]
# Questions:
# - Safe to pool releases?
#     - R5 is Val
# - Example for contrastChangeDetection available. How to preprocess others?
# - Relationship between release, task, and challenge parts?
# - Challenge part 1: (`batch`, `n_chans`, `n_times`) input -> (batch, 1) response time
# - Challenge part 2: (`batch`, `n_chans`, `n_times`) input -> (batch, 4) psychopathology scores
# - Why 129 channels?
# - What annotations are continuous v/s discrete?
# - How are p-factor, attention, internalizing, externalizing computed/recorded? Are these subject level, trial level or continuous?
# - Movie watching: how long do they watch?
#
# - Where we can be clever
#     - Training task formulation informed by the EEG collection tasks
#         - Masked EEG prediction
#         - Include task timing in input? (Model eval input format seems to be `batch`, `n_chans`, `n_times` so separate pretraining from challenge specific inference?)
#         - Pretraining with tasks other than the challenge tasks
#             - Add other derived annotations from raw data?
#                 - Eyes open/closed?
#                 - contrast values can be computed from raw data, increases linearly with time
#             - Training time multimodality? Single modality inference?
#         - Mapping EEG to Response time shouldn't be the primary modeling objective. Embedding should implicitly encode likelihood of response.
#     - Arch obviously
#         - Mamba SSM
#         - JEPA (available in braindecode)
#         - Labram (available in braindecode)
#         - TITANS / ATLAS
#         - Hierarchical Reasoning Model
#     - RL Reward that exploits EEG data collection protocol and what we know about EEG already
#     - Channel correlation for grouping
#

# %%
dataset_ccd_releases["R5"].datasets[0].raw
# %%
