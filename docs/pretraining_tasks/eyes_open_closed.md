# Eyes-Open / Eyes-Closed Proxy *(Tier 2)*

```mermaid
graph LR
    RestingState[Resting State window] --> HED[HED annotation\n(opened/closed)]
    HED --> Label[
      Binary label
    ]
    RestingState --> Encoder
    Encoder --> Classifier
    Classifier --> CE[Cross-entropy loss]
```

**Inputs**
- `RestingState` runs with `instructed_toOpenEyes` / `…CloseEyes` events.

**Objective**
- Lightweight supervised head predicting eyes-open vs. closed to stabilize alpha-band representation.

**Benefit**
- Acts as a regularizer; easy to train alongside self-supervised objectives.
