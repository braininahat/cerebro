# Feedback-Consistency Clustering *(Tier 2)*

```mermaid
graph TD
    A[CCD Trial embedding] -->|feedback = smiley| P[Positive cluster]
    A -->|feedback = sad_face| N[Negative cluster]
    P --> Loss[Margin loss]
    N --> Loss
```

**Description**
- Uses CCD `feedback` labels (`smiley_face` vs. `sad_face`).
- Applies an intra-subject margin loss encouraging consistent embeddings within the same feedback class.

**Purpose**
- Provides a weak ErrP-inspired signal that may refine error-related features.
- Only pursue after core Tier 1 objectives are stable.
