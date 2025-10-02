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
- Preprocessing policy: use competition-provided 0.5–50 Hz filtered data, apply per-recording z-score before windowing, no extra notch/ICA unless mirrored at inference.
- Extend datamodules to expose metadata needed for protocol-aware tasks (e.g., `stimulus_cond`, trial markers) while reproducing starter windowing (CCD 0.5–2.5 s; generic 4 s windows with 2 s stride).

### 3. Baseline Backbone Ladder (2–10 Oct)
- Goal: establish Challenge 2 baselines first using 4 s windows and simple subject-level pooling.
- Step 1: train Braindecode CNNs (`EEGNet`, `EEGTCNet`, `Deep4Net`, optionally `EEGNeX`) with mean pooling of window embeddings + demographics; record R5 nRMSE.
- Step 2: add `EEGConformer` only if attention cost fits 100 Hz inputs and the 20 GB inference constraint.
- Step 3: introduce the custom `EEGConvBackbone` (with optional Mamba/SSM block) only if baseline capacity saturates.

### 4. Self-/Unsupervised Pretraining (9–18 Oct)
- Train on R1–R4, R6–R11 (R5 held out) using short windows compatible with both challenges.
- Tier 1 priorities:
  1. Masked time×channel prediction (BENDR/JEPA-style) using the shared CNN backbone.
  2. SSVEP contrastive learning on surround-suppression data via CCA/PLV features (positives = same stimulus condition, negatives = others).
  3. Channel dropout + mild augmentations (time masking, jitter, Gaussian noise).
  4. CCD pre-trial RT regression head (direct supervised objective).
- Tier 2 (only if time allows): movie timestamp contrastive, feedback-consistency clustering, eyes-open/closed proxy, sequence-learning objectives.
- Evaluate via subject-wise CV on train releases; retain top checkpoints for fine-tuning.

### 5. Challenge 2 Fine-Tuning (15–24 Oct)
- Build per-subject datasets (≥15 min) with 4 s windows (2 s stride).
- Pooling ladder: start with mean pooling of window embeddings + demographics → MLP; escalate to attention/weighted pooling only if mean pooling saturates.
- Train 4-output regressor (p-factor, attention, internalizing, externalizing); monitor R5 nRMSE.
- Limit hyperparameter sweeps (≤3 runs per backbone); log compute usage.

### 6. Challenge 1 Adaptation (20–27 Oct)
- Reuse the best backbone from Challenge 2.
- Inputs: CCD pre-trial and 0.5–2.5 s windows; include demographics.
- Head: lightweight regressor; optionally add gradient-reversal for subject invariance.
- Integrate SSVEP features learned from SuS; ensure inference matches preprocessing (100 Hz, z-scored).
- Validate on R5 nRMSE.

### 7. Evaluation, Logging, Packaging (24–31 Oct)
- Maintain subject-disjoint split (train: R1–4,6–11; val: R5).
- Metrics: nRMSE per C2 factor + weighted score (0.3*C1 + 0.7*C2).
- Log available GPUs/storage and cap experiments accordingly.
- Package Codabench submission (models, scripts, env) after final dry-run inference.

### 8. Risk Controls
- No nested CV; rely on fixed train/validation split.
- Guard against protocol leakage (HED events for structure only; heads trained on allowed inputs).
- Respect availability flags; log excluded subjects/runs (e.g., corrupted BDF files).

### 9. Timeline to 31 Oct
- **By 10 Oct**: subject-level datamodule + EEGNet/Deep4Net baselines logged.
- **By 15 Oct**: masked modeling + SSVEP contrastive running; checkpoint backbone.
- **By 20 Oct**: Challenge 2 mean-pooling head tuned; R5 nRMSE recorded.
- **By 24 Oct**: Challenge 2 attention/advanced pooling (if needed); finalize backbone.
- **By 27 Oct**: Challenge 1 fine-tuning using shared backbone; record R5 nRMSE.
- **31 Oct**: Lock models/configs; submit Codabench package.
