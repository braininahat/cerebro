# Codebase Architecture for Multi-Phase Training

**Purpose**: Design a modular, extensible codebase that supports SignalJEPA multi-phase training while maintaining DRY principles, type safety, and ease of experimentation.

---

## Table of Contents
1. [Design Principles](#design-principles)
2. [Directory Structure](#directory-structure)
3. [Core Abstractions](#core-abstractions)
4. [Data Pipeline Architecture](#data-pipeline-architecture)
5. [Model Architecture](#model-architecture)
6. [Training Infrastructure](#training-infrastructure)
7. [Configuration System](#configuration-system)
8. [Tooling & Utilities](#tooling--utilities)
9. [Testing Strategy](#testing-strategy)
10. [Migration Path](#migration-path)

---

## Design Principles

### 1. **Composition Over Inheritance**
Favor composable modules over deep inheritance hierarchies.

**Bad** (rigid inheritance):
```python
class SignalJEPAModel(BaseModel):
    class WithContrastive(SignalJEPAModel):
        class WithAuxiliaryTasks(WithContrastive):
            ...  # Explosion of subclasses
```

**Good** (composition):
```python
model = Encoder(...)
model = with_contrastive_head(model, proj_dim=256)
model = with_auxiliary_heads(model, tasks=['age', 'sex', 'p_factor'])
```

### 2. **DRY with Parameterization**
Extract common patterns into configurable functions, not copy-paste.

**Example**: Masking strategies should share core logic
```python
def apply_mask(x, mask_fn, mask_ratio):
    """Generic masking for spatial/temporal/spatiotemporal."""
    mask = mask_fn(x.shape, mask_ratio)
    return x * mask, mask
```

### 3. **Explicit Over Implicit**
Configuration and data flow should be obvious from code.

**Bad**:
```python
model.train()  # What's being trained? What loss?
```

**Good**:
```python
train_signaljepa(
    model=encoder,
    data=tuh_hbn_mixed,
    loss=SignalJEPALoss(masking='spatiotemporal', aux_weight=0.45),
    ...
)
```

### 4. **Immutable Configs**
Use frozen dataclasses or Pydantic for configurations to prevent accidental mutation.

### 5. **Type Everything**
Extensive type hints for IDE support and early error detection.

```python
def create_windows(
    raw: mne.io.Raw,
    window_sec: float,
    stride_sec: float,
    sfreq: int
) -> torch.Tensor:
    ...
```

### 6. **Separation of Concerns**
- **Data**: Loading, preprocessing, windowing
- **Models**: Architecture definitions
- **Training**: Loss functions, optimizers, training loops
- **Evaluation**: Metrics, probing, visualization

---

## Directory Structure

```
cerebro/
├── __init__.py
├── cli/
│   ├── __init__.py
│   ├── train.py              # Lightning CLI entry point
│   └── evaluate.py           # Evaluation CLI
│
├── data/
│   ├── __init__.py
│   ├── base.py               # Abstract base classes
│   ├── loaders/
│   │   ├── __init__.py
│   │   ├── hbn.py            # HBN dataset loading
│   │   ├── tuh.py            # TUH dataset loading
│   │   └── mixed.py          # Mixed TUH+HBN loader
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── resampling.py     # Sampling rate alignment
│   │   ├── windowing.py      # Window extraction
│   │   └── normalization.py  # Z-score, robust scaling
│   ├── augmentation/
│   │   ├── __init__.py
│   │   ├── masking.py        # Spatial/temporal/spatiotemporal
│   │   ├── transforms.py     # Time warping, noise injection
│   │   └── contrastive.py    # Positive/negative pair creation
│   └── modules/
│       ├── __init__.py
│       ├── signaljepa.py     # SignalJEPA data module
│       ├── contrastive.py    # Contrastive data module
│       ├── supervised.py     # Challenge 1/2 data modules
│       └── demographics.py   # Auxiliary task data loading
│
├── models/
│   ├── __init__.py
│   ├── base.py               # Abstract base model
│   ├── components/
│   │   ├── __init__.py
│   │   ├── encoders/
│   │   │   ├── __init__.py
│   │   │   ├── transformer.py    # Transformer encoder
│   │   │   ├── mamba.py          # Mamba encoder
│   │   │   ├── fno.py            # FNO encoder
│   │   │   └── channel_adapter.py # Perceiver cross-attention
│   │   ├── predictors/
│   │   │   ├── __init__.py
│   │   │   ├── prelocal.py       # Pre-local predictor
│   │   │   ├── postlocal.py      # Post-local predictor
│   │   │   └── contextual.py     # Contextual predictor
│   │   ├── heads/
│   │   │   ├── __init__.py
│   │   │   ├── regression.py     # Generic regression head
│   │   │   ├── classification.py # Generic classification head
│   │   │   └── projection.py     # Contrastive projection head
│   │   └── aggregation/
│   │       ├── __init__.py
│   │       ├── mean_pool.py      # Mean pooling
│   │       └── attention_pool.py # Attention pooling
│   ├── signaljepa/
│   │   ├── __init__.py
│   │   ├── model.py          # Full SignalJEPA model
│   │   ├── target_encoder.py # EMA target encoder
│   │   └── auxiliary.py      # Auxiliary task heads
│   ├── contrastive/
│   │   ├── __init__.py
│   │   └── model.py          # Contrastive model wrapper
│   └── supervised/
│       ├── __init__.py
│       ├── challenge1.py     # Challenge 1 model
│       └── challenge2.py     # Challenge 2 model
│
├── losses/
│   ├── __init__.py
│   ├── signaljepa.py         # SignalJEPA reconstruction loss
│   ├── contrastive.py        # InfoNCE loss
│   ├── auxiliary.py          # Auxiliary task losses
│   └── composite.py          # Multi-loss combiner
│
├── training/
│   ├── __init__.py
│   ├── base.py               # Base Lightning module
│   ├── signaljepa.py         # Phase 1 training module
│   ├── contrastive.py        # Phase 2 training module
│   ├── supervised.py         # Phase 3 training module
│   └── multitask.py          # Multi-task training
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py            # NRMSE, MAE, accuracy
│   ├── probing.py            # Linear probing for evaluation
│   ├── isc.py                # Inter-subject correlation
│   └── visualization.py      # Plotting utilities
│
├── callbacks/
│   ├── __init__.py
│   ├── checkpoint_fix.py     # Existing checkpoint compatibility
│   ├── model_autopsy.py      # Existing model inspection
│   ├── phase_transition.py   # Phase 1 → 2 → 3 transition
│   └── evaluation_probe.py   # Continuous probing during training
│
├── utils/
│   ├── __init__.py
│   ├── logging.py            # Rich logging setup
│   ├── reproducibility.py    # Seed setting
│   ├── channel_mapping.py    # 21ch ↔ 128ch mappings
│   ├── demographics.py       # Demographic data utils
│   └── event_parsing.py      # Parse BIDS event files
│
└── configs/
    ├── base.yaml             # Global defaults
    ├── data/
    │   ├── hbn.yaml
    │   ├── tuh.yaml
    │   └── mixed.yaml
    ├── model/
    │   ├── encoder_transformer.yaml
    │   ├── encoder_mamba.yaml
    │   └── channel_adapter.yaml
    ├── phase1/
    │   ├── signaljepa_base.yaml
    │   ├── signaljepa_mini.yaml  # Fast prototyping
    │   └── signaljepa_full.yaml  # Full pretraining
    ├── phase2/
    │   ├── contrastive_movie.yaml
    │   ├── contrastive_resting.yaml
    │   └── contrastive_full.yaml
    ├── phase3/
    │   ├── supervised_challenge1.yaml
    │   ├── supervised_challenge2.yaml
    │   └── supervised_multitask.yaml
    └── ablations/
        ├── no_contrastive.yaml
        ├── no_auxiliary.yaml
        └── supervised_only.yaml
```

---

## Core Abstractions

### 1. Data Pipeline Components

#### Base Dataset Interface
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

@dataclass
class EEGWindow:
    """Single EEG window with metadata."""
    data: torch.Tensor  # (channels, time)
    subject_id: str
    task: str
    timestamp: float
    metadata: dict  # Flexible metadata (age, sex, etc.)

class EEGDataset(ABC):
    """Abstract base for all EEG datasets."""

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> EEGWindow:
        pass

    @property
    @abstractmethod
    def n_channels(self) -> int:
        """Number of channels in this dataset."""
        pass

    @property
    @abstractmethod
    def sfreq(self) -> float:
        """Sampling frequency in Hz."""
        pass

    @property
    @abstractmethod
    def channel_names(self) -> list[str]:
        """List of channel names."""
        pass
```

#### Masking Strategy Protocol
```python
from typing import Protocol

class MaskingStrategy(Protocol):
    """Protocol for masking strategies."""

    def __call__(
        self,
        shape: tuple[int, ...],  # (batch, channels, time)
        mask_ratio: float
    ) -> torch.Tensor:  # (batch, channels, time) boolean mask
        ...

# Implementations
class SpatialMasking:
    def __call__(self, shape, mask_ratio):
        # Random channel masking
        ...

class TemporalMasking:
    def __call__(self, shape, mask_ratio):
        # Random temporal block masking
        ...

class SpatiotemporalMasking:
    def __call__(self, shape, mask_ratio):
        # Combined 3D masking
        ...
```

#### Contrastive Pair Generator
```python
@dataclass
class ContrastivePair:
    anchor: torch.Tensor
    positive: torch.Tensor
    negatives: torch.Tensor  # (num_negatives, channels, time)
    anchor_metadata: dict
    positive_metadata: dict

class PairGenerator(ABC):
    @abstractmethod
    def generate_pairs(self, windows: list[EEGWindow]) -> list[ContrastivePair]:
        pass

class MovieISCGenerator(PairGenerator):
    def generate_pairs(self, windows):
        # Group by movie and timestamp
        # Create inter-subject positive pairs
        ...

class RestingStateGenerator(PairGenerator):
    def generate_pairs(self, windows):
        # Group by state (eyes open/closed)
        # Create state-based contrastives
        ...
```

### 2. Model Components

#### Encoder Factory
```python
from enum import Enum

class EncoderType(Enum):
    TRANSFORMER = "transformer"
    MAMBA = "mamba"
    FNO = "fno"
    HYBRID = "hybrid"

def create_encoder(
    encoder_type: EncoderType,
    n_channels: int,
    d_model: int,
    **kwargs
) -> nn.Module:
    """Factory for creating encoders."""
    if encoder_type == EncoderType.TRANSFORMER:
        return TransformerEncoder(n_channels, d_model, **kwargs)
    elif encoder_type == EncoderType.MAMBA:
        return MambaEncoder(n_channels, d_model, **kwargs)
    elif encoder_type == EncoderType.FNO:
        return FNOEncoder(n_channels, d_model, **kwargs)
    elif encoder_type == EncoderType.HYBRID:
        return HybridEncoder(n_channels, d_model, **kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
```

#### Composable Heads
```python
class HeadRegistry:
    """Registry pattern for heads."""
    _heads: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(head_cls):
            cls._heads[name] = head_cls
            return head_cls
        return decorator

    @classmethod
    def create(cls, name: str, input_dim: int, **kwargs):
        if name not in cls._heads:
            raise ValueError(f"Unknown head: {name}")
        return cls._heads[name](input_dim, **kwargs)

@HeadRegistry.register("regression")
class RegressionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

# Usage
age_head = HeadRegistry.create("regression", input_dim=512, output_dim=1)
sex_head = HeadRegistry.create("classification", input_dim=512, output_dim=2)
```

### 3. Training Infrastructure

#### Lightning Module Base
```python
import lightning as L

class BaseEEGModule(L.LightningModule):
    """Base class for all training modules."""

    def __init__(self, encoder: nn.Module, optimizer_config: dict):
        super().__init__()
        self.encoder = encoder
        self.optimizer_config = optimizer_config

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.optimizer_config,
            params=self.parameters()
        )
        return optimizer

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    def log_metrics(self, metrics: dict, prefix: str, prog_bar: bool = True):
        """Standardized metric logging."""
        for key, value in metrics.items():
            self.log(
                f"{prefix}/{key}",
                value,
                prog_bar=prog_bar,
                sync_dist=True
            )
```

#### Phase-Specific Modules
```python
class SignalJEPAModule(BaseEEGModule):
    def __init__(
        self,
        encoder: nn.Module,
        target_encoder: nn.Module,
        predictor: nn.Module,
        auxiliary_heads: dict[str, nn.Module],
        loss_config: dict,
        optimizer_config: dict
    ):
        super().__init__(encoder, optimizer_config)
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.auxiliary_heads = nn.ModuleDict(auxiliary_heads)
        self.loss_fn = hydra.utils.instantiate(loss_config)

    def training_step(self, batch, batch_idx):
        x, metadata = batch
        x_masked, mask = self.apply_masking(x)

        # Context encoding
        context = self.encoder(x_masked)

        # Target encoding (no grad)
        with torch.no_grad():
            target = self.target_encoder(x)

        # Prediction
        pred = self.predictor(context, mask)

        # Reconstruction loss
        loss_recon = self.loss_fn(pred, target)

        # Auxiliary losses
        loss_aux = 0
        aux_metrics = {}
        for name, head in self.auxiliary_heads.items():
            if name in metadata:
                pred_aux = head(context.mean(dim=-1))  # Pool over time
                loss_aux_i = F.mse_loss(pred_aux.squeeze(), metadata[name])
                loss_aux += loss_aux_i
                aux_metrics[f"aux_{name}"] = loss_aux_i

        # Total loss
        loss = loss_recon + self.loss_fn.aux_weight * loss_aux

        # Logging
        self.log_metrics({
            "loss": loss,
            "loss_recon": loss_recon,
            "loss_aux": loss_aux,
            **aux_metrics
        }, prefix="train")

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # EMA update for target encoder
        self.update_target_encoder()

    def update_target_encoder(self):
        """Momentum update of target encoder."""
        for param_q, param_k in zip(
            self.encoder.parameters(),
            self.target_encoder.parameters()
        ):
            param_k.data = (
                self.loss_fn.ema_momentum * param_k.data +
                (1 - self.loss_fn.ema_momentum) * param_q.data
            )
```

---

## Data Pipeline Architecture

### Unified Data Loading

#### Mixed Dataset
```python
class MixedDataset(torch.utils.data.Dataset):
    """Mix TUH and HBN datasets with configurable sampling."""

    def __init__(
        self,
        tuh_dataset: EEGDataset,
        hbn_dataset: EEGDataset,
        tuh_weight: float = 0.3,
        resample_to_hz: int = 100
    ):
        self.tuh = tuh_dataset
        self.hbn = hbn_dataset
        self.tuh_weight = tuh_weight

        # Resample TUH to match HBN
        self.tuh = ResampledDataset(self.tuh, target_hz=resample_to_hz)

        # Calculate sampling probabilities
        self.dataset_probs = [tuh_weight, 1 - tuh_weight]
        self.datasets = [self.tuh, self.hbn]

    def __len__(self):
        return len(self.tuh) + len(self.hbn)

    def __getitem__(self, idx):
        # Sample dataset based on weights
        dataset = np.random.choice(self.datasets, p=self.dataset_probs)
        # Sample random index from chosen dataset
        idx = np.random.randint(len(dataset))
        return dataset[idx]
```

#### Channel Adaptation Layer
```python
class ChannelAdapter(nn.Module):
    """Perceiver-style channel adaptation."""

    def __init__(
        self,
        max_channels: int = 128,
        d_model: int = 256,
        num_heads: int = 8
    ):
        super().__init__()
        # Fixed learnable electrode queries
        self.electrode_queries = nn.Parameter(
            torch.randn(max_channels, d_model)
        )

        # Per-channel encoder
        self.channel_encoder = nn.Conv1d(1, d_model, kernel_size=3, padding=1)

        # Cross-attention: queries attend to observed channels
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_channels_observed, time)
        Returns:
            (batch, 128, d_model)  # Always 128 channels
        """
        B, C, T = x.shape

        # Encode each observed channel
        channel_tokens = []
        for c in range(C):
            token = self.channel_encoder(x[:, c:c+1, :])  # (B, d_model, T)
            token = token.mean(dim=-1)  # Pool over time: (B, d_model)
            channel_tokens.append(token)
        keys_values = torch.stack(channel_tokens, dim=1)  # (B, C, d_model)

        # Broadcast queries to batch
        queries = self.electrode_queries.unsqueeze(0).expand(B, -1, -1)

        # Cross-attention
        output, _ = self.cross_attn(queries, keys_values, keys_values)
        return output  # (B, 128, d_model)
```

---

## Configuration System

### Hydra Composition

**Base config** (`configs/base.yaml`):
```yaml
defaults:
  - _self_
  - data: ???  # Must be specified
  - model: ???  # Must be specified
  - training: ???  # Must be specified

seed: 42
device: cuda
output_dir: outputs/${now:%Y-%m-%d_%H-%M-%S}

wandb:
  enabled: true
  project: cerebro-signaljepa
  entity: ${oc.env:WANDB_ENTITY}
```

**Phase 1 config** (`configs/phase1/signaljepa_full.yaml`):
```yaml
defaults:
  - /base
  - /data/mixed
  - /model/encoder_transformer
  - _self_

data:
  tuh_weight: 0.3
  hbn_releases: [R1, R2, R3, R4, R6, R7, R8, R9, R10, R11]
  window_sec: 4.0
  stride_sec: 2.0
  resample_to_hz: 100

model:
  encoder:
    d_model: 512
    n_layers: 12
    n_heads: 8
  predictor:
    type: prelocal
    d_model: 512
  auxiliary_heads:
    age: {type: regression, output_dim: 1}
    sex: {type: classification, output_dim: 2}
    p_factor: {type: regression, output_dim: 1}
    attention: {type: regression, output_dim: 1}
    internalizing: {type: regression, output_dim: 1}
    externalizing: {type: regression, output_dim: 1}

loss:
  masking_type: spatiotemporal
  mask_ratio: 0.6
  aux_weight: 0.45
  ema_momentum: 0.996

training:
  optimizer:
    _target_: torch.optim.AdamW
    lr: 1.0e-4
    weight_decay: 0.05
  scheduler:
    _target_: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    T_0: 10
    T_mult: 2
  batch_size: 256
  epochs: 200
  gradient_clip_val: 1.0

callbacks:
  - _target_: cerebro.callbacks.evaluation_probe.EvaluationProbe
    probe_every_n_epochs: 5
  - _target_: lightning.pytorch.callbacks.ModelCheckpoint
    monitor: val/loss
    mode: min
    save_top_k: 3
```

---

## Tooling & Utilities

### 1. Experiment Management

#### WandB Integration
```python
class WandBLogger:
    """Custom WandB logger with phase tracking."""

    def __init__(self, config: dict, phase: str):
        self.run = wandb.init(
            project=config['project'],
            entity=config['entity'],
            tags=[phase],
            config=config
        )

    def log_phase_transition(self, from_phase: str, to_phase: str, checkpoint: str):
        """Log phase transitions."""
        wandb.log({
            "phase_transition": {
                "from": from_phase,
                "to": to_phase,
                "checkpoint": checkpoint
            }
        })

    def log_ablation(self, ablation_name: str, metric_diff: float):
        """Log ablation study results."""
        wandb.log({
            f"ablation/{ablation_name}": metric_diff
        })
```

### 2. Checkpoint Management

#### Phase Checkpoint Handler
```python
class PhaseCheckpointManager:
    """Manage checkpoints across training phases."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.phase_checkpoints = {
            "phase1": checkpoint_dir / "phase1_best.ckpt",
            "phase2": checkpoint_dir / "phase2_best.ckpt",
            "phase3": checkpoint_dir / "phase3_best.ckpt"
        }

    def save_phase_checkpoint(self, phase: str, module: L.LightningModule):
        """Save checkpoint for a phase."""
        path = self.phase_checkpoints[phase]
        trainer.save_checkpoint(path)
        logger.info(f"Saved {phase} checkpoint to {path}")

    def load_for_next_phase(
        self,
        current_phase: str,
        next_phase_module: L.LightningModule
    ) -> L.LightningModule:
        """Load checkpoint and prepare for next phase."""
        checkpoint_path = self.phase_checkpoints[current_phase]

        # Load encoder weights
        checkpoint = torch.load(checkpoint_path)
        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("encoder.")
        }
        next_phase_module.encoder.load_state_dict(encoder_state)

        logger.info(f"Loaded {current_phase} encoder for {next_phase_module.__class__.__name__}")
        return next_phase_module
```

### 3. Reproducibility

```python
def set_seed(seed: int):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    lightning.seed_everything(seed)
```

### 4. Channel Mapping Utilities

```python
@dataclass
class ElectrodeMapping:
    """Maps between different electrode systems."""
    standard_1020_21 = [
        "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
        "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
        "Fz", "Cz", "Pz", "A1", "A2"
    ]

    standard_1020_128 = [...]  # Full 128-channel list

    @classmethod
    def get_overlap_indices(cls) -> dict[str, int]:
        """Get mapping from 21-channel to 128-channel indices."""
        mapping = {}
        for i, ch in enumerate(cls.standard_1020_21):
            if ch in cls.standard_1020_128:
                mapping[i] = cls.standard_1020_128.index(ch)
        return mapping

    @classmethod
    def pad_to_128(cls, x: torch.Tensor) -> torch.Tensor:
        """Pad 21-channel data to 128 channels with zeros."""
        B, C, T = x.shape
        assert C == 21, f"Expected 21 channels, got {C}"

        x_padded = torch.zeros(B, 128, T, device=x.device)
        mapping = cls.get_overlap_indices()

        for src_idx, dst_idx in mapping.items():
            x_padded[:, dst_idx, :] = x[:, src_idx, :]

        return x_padded
```

---

## Testing Strategy

### Unit Tests
```python
# tests/test_masking.py
def test_spatial_masking():
    masking = SpatialMasking()
    x = torch.randn(32, 128, 400)
    mask = masking(x.shape, mask_ratio=0.6)

    assert mask.shape == x.shape
    assert 0.5 < mask.sum() / mask.numel() < 0.7  # ~60% masked
    assert mask.dtype == torch.bool

# tests/test_channel_adapter.py
def test_channel_adapter_tuh():
    adapter = ChannelAdapter(max_channels=128, d_model=256)
    x_tuh = torch.randn(16, 21, 400)  # TUH: 21 channels
    out = adapter(x_tuh)

    assert out.shape == (16, 128, 256)  # Always 128 channels

def test_channel_adapter_hbn():
    adapter = ChannelAdapter(max_channels=128, d_model=256)
    x_hbn = torch.randn(16, 128, 400)  # HBN: 128 channels
    out = adapter(x_hbn)

    assert out.shape == (16, 128, 256)
```

### Integration Tests
```python
# tests/test_phase1_training.py
def test_signaljepa_training_step():
    module = SignalJEPAModule(...)
    batch = (torch.randn(8, 128, 400), {"age": torch.tensor([25, 30, ...])})

    loss = module.training_step(batch, batch_idx=0)

    assert loss.requires_grad
    assert loss.item() > 0
    assert "train/loss_recon" in module.trainer.logged_metrics
    assert "train/aux_age" in module.trainer.logged_metrics
```

### End-to-End Tests
```python
# tests/test_full_pipeline.py
@pytest.mark.slow
def test_full_pipeline():
    """Test Phase 1 → 2 → 3 pipeline."""
    # Phase 1
    phase1_module = train_phase1(epochs=2)
    checkpoint_manager.save_phase_checkpoint("phase1", phase1_module)

    # Phase 2
    phase2_module = prepare_phase2(checkpoint_manager)
    train_phase2(phase2_module, epochs=2)
    checkpoint_manager.save_phase_checkpoint("phase2", phase2_module)

    # Phase 3
    phase3_module = prepare_phase3(checkpoint_manager)
    metrics = train_phase3(phase3_module, epochs=2)

    assert metrics["val/nrmse_c1"] < 1.0
    assert metrics["val/nrmse_c2"] < 1.0
```

---

## Migration Path

### Step 1: Create New Branch
```bash
git checkout main
git pull
git checkout -b feat/signaljepa-pipeline
```

### Step 2: Scaffold New Structure (Session 1)
```bash
# Create directory structure
mkdir -p cerebro/{data/{loaders,preprocessing,augmentation},models/components/{encoders,predictors,heads}}

# Create __init__.py files
find cerebro -type d -exec touch {}/__init__.py \;

# Move existing code
mv cerebro/data/hbn.py cerebro/data/loaders/
mv cerebro/models/challenge1.py cerebro/models/supervised/
```

### Step 3: Implement Core Abstractions (Session 1-2)
- [ ] `data/base.py`: EEGWindow, EEGDataset
- [ ] `data/augmentation/masking.py`: Masking strategies
- [ ] `models/components/channel_adapter.py`: Channel adaptation
- [ ] `training/base.py`: BaseEEGModule

### Step 4: Implement Phase 1 (Session 3-4)
- [ ] SignalJEPA encoder, predictor, auxiliary heads
- [ ] Mixed dataset (TUH + HBN)
- [ ] Training module
- [ ] Config files

### Step 5: Implement Phase 2 (Session 5-6)
- [ ] Contrastive pair generators
- [ ] InfoNCE loss
- [ ] Contrastive training module

### Step 6: Implement Phase 3 (Session 7)
- [ ] Challenge 1/2 supervised modules
- [ ] Fine-tuning pipeline

### Step 7: Evaluation & Iteration (Session 8+)
- [ ] End-to-end testing
- [ ] Ablation studies
- [ ] Performance optimization

---

## Key Benefits of This Architecture

### 1. **Modularity**
Each component (encoder, predictor, head, loss) is independent and swappable.

### 2. **Extensibility**
Adding new encoders, masking strategies, or tasks requires minimal changes.

### 3. **DRY**
Common patterns (loss composition, logging, checkpointing) abstracted into reusable components.

### 4. **Type Safety**
Extensive type hints catch errors early and improve IDE support.

### 5. **Testability**
Small, focused modules are easy to unit test.

### 6. **Reproducibility**
Immutable configs and seed management ensure reproducible experiments.

### 7. **Clarity**
Explicit data flow and configuration make the pipeline easy to understand.

---

## Future Extensions

### 1. Multi-GPU Training
```python
# Already supported via Lightning
trainer = L.Trainer(
    devices=2,
    strategy="ddp",
    precision="16-mixed"
)
```

### 2. Hyperparameter Tuning
```python
# Optuna integration
def objective(trial):
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    mask_ratio = trial.suggest_uniform("mask_ratio", 0.4, 0.8)

    config = {
        "training": {"lr": lr},
        "loss": {"mask_ratio": mask_ratio}
    }

    trainer = L.Trainer(...)
    module = SignalJEPAModule(config)
    trainer.fit(module)

    return trainer.callback_metrics["val/loss"].item()

study = optuna.create_study()
study.optimize(objective, n_trials=50)
```

### 3. Distributed Data Loading
```python
# Use WebDataset for large-scale datasets
from webdataset import WebLoader

dataset = wds.WebDataset("s3://bucket/tuh-{000..100}.tar")
    .decode()
    .to_tuple("eeg.pth", "metadata.json")
```

---

**Document Version**: 1.0
**Last Updated**: 2025-10-27
**Author**: Varun (with Claude assistance)
