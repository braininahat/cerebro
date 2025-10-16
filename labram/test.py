import torch
B, N, T = 8, 129, 200
x = torch.randn(B, N, T, device='cuda', dtype=torch.float32) * 1e-3
x = x.contiguous()
try:
    y = torch.fft.fft(x, n=T, dim=-1)
    print("GPU cuFFT OK:", y.shape)
except RuntimeError as e:
    print("GPU cuFFT FAILED:", e)
    y = torch.fft.fft(x.cpu(), n=T, dim=-1).to('cuda')
    print("CPU FFT fallback OK:", y.shape)
