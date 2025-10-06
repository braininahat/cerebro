# Movie Timestamp Contrastive Learning *(Tier 2)*

```mermaid
flowchart LR
    subgraph Subject A
        A1[Window t0]
    end
    subgraph Subject B
        B1[Window t0]
    end
    subgraph Subject C
        C1[Window t_random]
    end
    A1 --> EncA[Encoder]
    B1 --> EncB[Encoder]
    C1 --> EncC[Encoder]
    EncA --> Loss[InfoNCE]
    EncB --> Loss
    EncC -. Negative .-> Loss
```

**Inputs**
- Windows cropped at identical offsets relative to `video_start` (no per-scene markers).
- Negatives sampled from different timestamps or other movies.

**Objective**
- Pull embeddings of synchronized scenes together across subjects; push non-aligned windows apart.

**Use cases**
- Improves inter-subject coordination useful for Challenge 2 generalization.
