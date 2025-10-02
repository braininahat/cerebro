**Foundation-Model Roadmap for EEG2025**  
*Protocol-aware revision — 2 Oct 2025*

This plan bakes task/trial structure (HED events, timing, SSVEP stimulation) into pretraining while keeping preprocessing minimal. It aligns with the NeurIPS EEG Foundation Challenge timeline (final phase ends 31 Oct 2025).

---

### 0. Repo & Branch Conventions (Day 0)
- Work on branch `foundation-model` (or `feature/fm-protocol-aware`).
- Keep this roadmap mirrored in `AGENTS.md` at repo root.
- Add new top-level directories for long-term development: `fm/`, `fm/datamodules/`, `fm/tasks/`, `fm/models/`, `fm/eval/`, `fm/config/`, `scripts/submit/`.

### 1. Understand the Challenge & Dataset (Today)
- **Data**: 128-channel, 100 Hz EEG with six tasks (Resting State, Surround Suppression, Movie Watching, Sequence Learning, CCD, Symbol Search); BIDS + HED annotations; demographics; four psychopathology factors with availability flags.
- **Splits**: Releases R1–R11 for train/validation (subject-disjoint; R5 reserved for validation). Release R12 is hidden test (code submission).
- **Tasks**:
  - Challenge 1 (Transfer): predict CCD response time from SuS + pre-trial EEG; evaluate 0.5–2.5 s post-onset windows + demographics; metric nRMSE.
  - Challenge 2 (Psychopathology): per-subject regression of four factors; ≥15 min EEG per subject; metric nRMSE (70 % weight in final score).

### 2. Set Up the Project (2–4 Oct)
- Ensure loaders read BIDS/HED via `EEGChallengeDataset` or custom datamodules, including demographics and availability flags.
- Cache R1–R4 and R6–R11 mini/full releases (exclude R5 from training).
- Preprocessing policy: per-window/recording z-score; optional 0.5–50 Hz band-pass + 60 Hz notch; avoid ICA/ASR.
- Reproduce starter windowing for CCD (0.5–2.5 s) and generic 4 s windows with 2 s stride.

### 3. Backbone Architecture (2–8 Oct)
- **Stem**: 2‑D conv to downsample to ~25 Hz and mix local channels.
- **Sequence core**: Mamba-style SSM or EEG-Conformer blocks with RMSNorm + gated FFNs.
- **Spatial mixing**: lightweight learned channel mixing; optional correlation-based grouping.
- **Pooling**: global average or attention → 256 D embedding; variable-length friendly.
- **Demographics**: FiLM or concatenated embeddings for age/sex/handedness ahead of task heads.

### 4. Self-/Unsupervised Pretraining (2–14 Oct)
- Pretrain on R1–R4, R6–R11 across all tasks (passive + active); rely on HED events and timing cues.
- Objectives:
  - Masked time×channel modeling (MTCM).
  - JEPA latent prediction over HED-aligned context/target segments.
  - CPC/InfoNCE on adjacent segments (pre→post onset, etc.).
  - SSVEP-aware contrastive clustering using protocol tokens (flicker state bins).
  - Pre-trial readiness ranking (contrastive RT quantiles).
  - Feedback-consistency (ErrP proxy) with margin loss.
  - Optional eyes-open/closed proxy from Resting State.
- Augmentations: mild time masking, jitter, noise, ≤10 % channel dropout, same-subject mixup.
- Track validation via subject-wise CV; save top-k backbones.

### 5. Challenge 1 Fine-Tuning (8–18 Oct)
- Inputs: CCD 0.5–2.5 s windows (optionally add pre-trial summary token) + demographics.
- Head: small MLP regressor; gradient-reversal adversary for subject invariance.
- Train on R1–R4, R6–R11; early-stop on R5 nRMSE.
- Ablations: pretraining objective mix, pre-trial embedding, feedback head, spatial grouping.

### 6. Challenge 2 Fine-Tuning (12–22 Oct)
- Build per-subject window sets (≥15 min total EEG) across tasks.
- Set encoder: transformer/DeepSets to pool windows; fuse demographics.
- Head: 4-output regressor for p-factor, attention, internalizing, externalizing.
- Freeze backbone partially; monitor nRMSE per factor and weighted score on R5.

### 7. Evaluation, Logging, Packaging (18–27 Oct)
- Validation discipline: subject-disjoint; R5 only for model selection.
- Metrics: nRMSE for C1 and each C2 factor (0.3/0.7 weighting).
- Hyperparameter sweeps: LR, batch, dropout, mask ratios (limited budget).
- Package Codabench submission (models + scripts + env) after dry-run inference.

### 8. Optional Last-Mile RL (24–29 Oct)
- Reward-based objective scheduler (bandit) using R5 proxy nRMSE.
- Curriculum over subjects/windows targeting hard pre-trial→RT cases.
- Keep RL thin and decoupled from backbone.

### 9. Risk Controls
- Nested CV across train releases to avoid R5 overfitting.
- Protocol leakage guard: use HED events for structure only; final heads trained strictly on allowed inputs.
- Respect availability flags; ablate inclusion of “Caution” runs.

### 10. Timeline to 31 Oct
- **By 7 Oct**: BIDS/HED datamodule + MTCM & JEPA running; initial backbone checkpoint.
- **By 14 Oct**: CPC, SSVEP-aware contrastive, readiness, and feedback heads implemented; top-2 backbones selected.
- **By 18 Oct**: Challenge 1 head tuned; R5 nRMSE logged; initial ablations.
- **By 22 Oct**: Challenge 2 set encoder trained; per-factor nRMSE reported.
- **By 27 Oct**: Lock hyperparameters; prepare Codabench package; dry-run inference.
- **31 Oct**: Final code submission.
