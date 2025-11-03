#%%
from eegdash import EEGChallengeDataset


#%%
r5_ccd = EEGChallengeDataset(release="R5", cache_dir="/home/varun/repos/cerebro/data", mini=True, task="contrastChangeDetection")