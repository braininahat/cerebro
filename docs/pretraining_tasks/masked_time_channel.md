# Masked Time×Channel Prediction (JEPA-style)

```mermaid
flowchart LR
    subgraph Input
        A[Raw EEG window\n(Channels × Time)]
    end
    A --> B[Masker\n(drop contiguous time patches)]
    B --> C[Encoder f(·)]
    A --> D[Target extractor g(·)\n(on masked spans)]
    C --> E[Context embedding]
    D --> F[Target embedding]
    E --> G[JEPA loss\n(InfoNCE over positives/negatives)]
    F --> G
    G --> H[Backprop gradients]
```

**Formulation**
- Input windows keep native 100 Hz sampling and all 129 channels.
- Mask contiguous time segments (e.g., 100 ms) across all channels.
- Context encoder `f(·)` sees masked window; target encoder `g(·)` processes the held-out span (teacher-student or EMA weights).
- InfoNCE / JEPA loss pulls context and true target embeddings together while pushing negatives (random spans) apart.

**Supervision**: self-supervised via reconstruction of latent representations.

**Compatibility**: Works on any task (SuS, CCD, movie, rest) with consistent window sizes.
