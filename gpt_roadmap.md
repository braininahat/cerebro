# Foundation-Model Roadmap for EEG2025

This document outlines the long-term plan for building a foundation-style solution that ultimately addresses both EEG2025 challenges. Timelines are indicative; revisit and adjust as progress evolves.

## 1. Understand the Challenge & Dataset (Week 1)

- **Challenge structure**: review both tasks in `startkit/README.md` and the official site. Challenge 1 predicts response time (RT) from Contrast Change Detection (CCD) epochs. Challenge 2 predicts the subject-level psychopathology (p-factor). Both operate on 129-channel, 100 Hz EEG recordings distributed in BIDS format.
- **Tasks available**: passive (Resting State, Surround Suppression, Movie Watching) and active (CCD, Sequence Learning, Symbol Search) recordings span 3 000+ subjects.
- **Window conventions**: Challenge 1 uses 2 s epochs starting 0.5 s post-stimulus; Challenge 2 aggregates 4 s windows with a 2 s stride, allowing random 2 s crops during training. Metrics emphasise MAE (C1) and concordance correlation (C2).

## 2. Set Up the Project (Week 1)

- Fork or clone the repo and work on a dedicated branch (e.g., `foundation-model`).
- Install dependencies: `pip install -r startkit/requirements.txt` plus project extras. Verify `EEGChallengeDataset` loads CCD, SuS, and other tasks.
- Download at least one mini release (R5 for validation) in the 100 Hz / 129-channel format.
- Run `notebooks/002_benchmark.py` to familiarise with existing architectures and baseline performance.
- Build a reusable data pipeline using Braindecode helpers (`create_windows_from_events`, `create_fixed_length_windows`), applying minimal preprocessing (per-epoch z-scoring, optional 0.5–50 Hz bandpass, notch). Avoid aggressive subject-specific cleaning to preserve cross-subject structure.

## 3. Design the Backbone Architecture (Weeks 2–3)

- **Conv–SSM/Conformer backbone**: start with a 2‑D convolutional stem to downsample from 100 Hz (e.g., to 25 Hz), followed by state-space (Mamba) or Conformer blocks for long-range temporal modelling. Keep optional depthwise spatial convolutions lightweight.
- **Latent embedding & pooling**: after sequence blocks, apply global average or attention pooling to produce a fixed embedding (≈256 D). Support variable-length inputs (2 s or 4 s windows).
- **Demographics embedding**: encode age, sex, handedness via small learnable embeddings; combine with backbone features via concatenation or FiLM modulation before regression heads.

## 4. Unsupervised / Self-Supervised Pretraining (Weeks 3–7)

Organisers encourage pretraining on passive tasks before fine-tuning.

- **Latent future prediction (JEPA)**: split windows into context/target segments; predict target latents using an energy-based contrastive loss.
- **Contrastive Predictive Coding (CPC)**: sample positives from adjacent segments within a recording, negatives across time/subjects; optimise InfoNCE.
- **Masked time–channel modelling**: mask patches in the time×channel plane, reconstruct latent codes to capture local dependencies.
- **Auxiliary EEG heads**: add lightweight supervised objectives (e.g., eyes open/closed, CCD feedback correctness) leveraging existing annotations.
- **Augmentations**: time masking, channel dropout, jitter, Gaussian noise, same-subject mixup; avoid heavy filtering to maintain generality.
- Cycle through all tasks and subjects (excluding held-out R5) with batch sizes 32–64. Monitor held-out loss, save backbone checkpoints once converged.

## 5. Challenge 1 Fine-Tuning (Weeks 7–8)

- Extract 2 s CCD epochs starting 0.5 s after stimulus onset (see `startkit/challenge_1.py`). Optionally add pre-stimulus context tokens.
- Attach an MLP regression head; concatenate demographic embeddings. Use MSE and consider a gradient-reversal adversarial head to discourage subject leakage.
- Fine-tune the head and optionally top backbone blocks with early stopping on a held-out R5 validation split (no R5 subjects in training). Track MAE, RMSE, nRMSE.
- Ablate: CCD-only vs CCD + context vs CCD + pretrained backbone; explore adding SuS as auxiliary supervision.

## 6. Challenge 2 Fine-Tuning (Weeks 8–9)

- Aggregate all 4 s windows (≥15 min per subject) across tasks. Construct per-subject window sets.
- Implement a transformer / set encoder to pool variable-length window sets into subject embeddings; append demographic embeddings.
- Train a regression head mapping to the p-factor (single output for current rules; keep architecture flexible). Freeze most of the backbone; optimise nRMSE or CCC. Validate on R5, and test multi-task fine-tuning alongside the Challenge 1 head.

## 7. Evaluation & Iteration (Week 10)

- Perform leave-one-mini-release-out cross-validation and record metrics per fold.
- Hyperparameter sweeps: learning rates, batch sizes, dropout (use W&B or equivalent).
- Compare against Braindecode baselines (EEGNet, Deep4Net, BIOT) on identical splits; document performance and efficiency differences.
- Conduct ablations for each pretraining objective and augmentation to identify impactful components.

## 8. Optional Last-Mile RL Enhancements (Week 11+)

After solid foundation performance:

- Explore reward-driven feature selection (e.g., MARS agents using CCD response-time rewards) layered atop the backbone.
- Test RL-based sample selection policies while ensuring they maintain generalisation.

## Summary

Following this roadmap will deliver a foundation model that leverages passive-task pretraining and supports both challenges. Once a robust baseline is established, reinforcement-learning enhancements can be treated as optional experiments rather than core dependencies.
