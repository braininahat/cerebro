import torch
import torch.backends.cuda as cuda_back
from eegdash.dataset import EEGChallengeDataset
import os

cuda_back.cufft_plan_cache.max_size = 0
B, N, T = 8, 129, 200
x = torch.randn(B, N, T, device='cuda', dtype=torch.float32) * 1e-3
x = x.contiguous()

DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.getcwd(), "data"))
dataset_ccd = EEGChallengeDataset(
    task="contrastChangeDetection",
    release="R5",
    cache_dir=DATA_DIR,
    mini=False
)

open()

try:
    y = torch.fft.fft(x, n=T, dim=-1)
    print("GPU cuFFT OK:", y.shape)
except RuntimeError as e:
    print("GPU cuFFT FAILED:", e)
    y = torch.fft.fft(x.cpu(), n=T, dim=-1).to('cuda')
    print("CPU FFT fallback OK:", y.shape)
