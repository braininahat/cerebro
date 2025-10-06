# SSVEP Frequency/Phase Contrastive Learning

```mermaid
flowchart LR
    subgraph SuS/CCD windows
        W1[Window i\nstimulus_cond = a]
        W2[Window j\nstimulus_cond = a]
        W3[Window k\nstimulus_cond = b]
    end
    W1 --> CCA1[CCA/PLV feature extractor]
    W2 --> CCA2[CCA/PLV feature extractor]
    W3 --> CCA3[CCA/PLV feature extractor]
    CCA1 --> E1[Encoder]
    CCA2 --> E2[Encoder]
    CCA3 --> E3[Encoder]
    E1 --> L[Contrastive Loss]\nclassDef pos fill:#c3f9c7;
    E2 --> L
    E3 -. Negative .-> L
```

**Formulation**
- Use surround-suppression (`stim_ON`, `stimulus_cond`) and CCD flicker events.
- Extract frequency-locked features via CCA/PLV at condition-specific harmonics (e.g., 15 Hz, 30 Hz, 45 Hz).
- Encode features and train with NT-Xent / InfoNCE:
  - Positives: windows sharing the same stimulus condition.
  - Negatives: windows with different conditions.

**Supervision**: Protocol-aware, label-free (stimulus condition derived from HED annotations).

**Outputs**: Condition-invariant SSVEP embeddings that transfer to CCD RT prediction.
