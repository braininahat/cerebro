# Channel Dropout Augmentation

```mermaid
flowchart LR
    A[Input window\n129 channels] --> B[Randomly sample dropout mask\np_drop ≤ 0.1]
    B --> C[Zero/Noise inject dropped channels]
    C --> D[Encoder]
    D --> E[Shared objective (e.g., JEPA, contrastive)]
```

**Purpose**
- Encourages robustness to missing or noisy electrodes.
- Applied inline with other objectives: masked modeling, SSVEP contrastive, supervised heads.

**Configuration**
- Drop probability ≤ 10 %; replace dropped channels with zeros or Gaussian noise.
- Maintain same preprocessing (z-score per recording) to avoid leakage.
