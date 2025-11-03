# Branch Comparison: docs/multi-phase-training-plan vs feat/varun/movie_contrastive

**Date**: 2025-10-27
**Purpose**: Document differences to ensure no functionality is lost during architecture refactor

---

## Summary

**Current Branch** (`docs/multi-phase-training-plan`): Has comprehensive TUH data modules and full-featured HBNDataModule (1061 lines)

**Movie Branch** (`feat/varun/movie_contrastive`): Has modular architecture (encoders, trainers, specialized data modules)

**Strategy**: Combine best of both - keep TUH modules from current, adopt architectural components from movie

---

## Files ONLY in Current Branch (MUST PRESERVE)

### Critical TUH Implementation
| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `cerebro/data/tuh_edf.py` | 1327 | TUH EDF data module with Zarr caching | **KEEP** |
| `cerebro/data/tuh.py` | 557 | TUH HDF5 data module | **KEEP** |
| `configs/datamodule/tuh.yaml` | 24 | TUH HDF5 config | **KEEP** |
| `configs/datamodule/tuh_edf.yaml` | 55 | TUH EDF config | **KEEP** |
| `configs/datamodule/tuh_edf_test.yaml` | 41 | TUH test config | **KEEP** |
| `configs/labram/README_TUH_EDF.md` | 277 | TUH documentation | **KEEP** |
| `configs/labram/codebook_tuh.yaml` | 156 | TUH codebook config | **KEEP** |
| `configs/labram/codebook_tuh_edf.yaml` | 191 | TUH EDF codebook config | **KEEP** |
| `configs/tuh_quick_test.yaml` | 164 | TUH quick test config | **KEEP** |

**Total TUH-related code**: ~2,792 lines
**Recent commits**: Zarr cache fixes, interrupt handling, corrupt cache handling

### Full-Featured HBN Data Module
| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `cerebro/data/hbn.py` | 1061 | Full HBNDataModule with challenge1/challenge2/pretrain modes | **MERGE** |

**Note**: Movie branch hbn.py is only 177 lines (backward compat dispatcher)

### Old-Style Configs (Backward Compatibility)
| File | Purpose | Status |
|------|---------|--------|
| `configs/challenge1_eegnex.yaml` | EEGNeX Challenge 1 (old style) | **PORT to new style** |
| `configs/challenge1_eegnex_mini.yaml` | EEGNeX Challenge 1 mini (old style) | **PORT to new style** |
| `configs/challenge1_jepa.yaml` | SignalJEPA Challenge 1 (old style) | **PORT to new style** |
| `configs/challenge1_jepa_mini.yaml` | SignalJEPA Challenge 1 mini (old style) | **PORT to new style** |
| `configs/challenge1_submission.yaml` | Challenge 1 submission (old style) | **PORT to new style** |

---

## Files ONLY in Movie Branch (TO ADOPT)

### Core Architecture Components
| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `cerebro/models/architectures.py` | 500 | RegressorModel, ContrastiveModel, MultitaskModel, etc. | **ADOPT** |
| `cerebro/models/builders.py` | 152 | Registry pattern for encoders | **ADOPT** |
| `cerebro/models/components/__init__.py` | 49 | Component exports | **ADOPT** |
| `cerebro/models/components/encoders.py` | 226 | BaseEncoder, EEGNeXEncoder, SignalJEPAEncoder | **ADOPT** |
| `cerebro/models/components/decoders.py` | 167 | RegressionHead, ProjectionHead, etc. | **ADOPT** |
| `cerebro/models/components/jepa_components.py` | 280 | JEPAEncoder, MambaEncoder | **ADOPT** |
| `cerebro/models/components/jepa_predictors.py` | 220 | JEPA predictors (trait/state/event) | **ADOPT** |

**Total architecture**: ~1,594 lines

### Trainer Modules
| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `cerebro/trainers/__init__.py` | 17 | Trainer exports | **ADOPT** |
| `cerebro/trainers/supervised.py` | 325 | SupervisedTrainer (model-agnostic) | **ADOPT** |
| `cerebro/trainers/contrastive.py` | 345 | ContrastiveTrainer (InfoNCE loss) | **ADOPT** |
| `cerebro/trainers/jepa.py` | 318 | JEPAPhase1Trainer | **ADOPT** |

**Total trainers**: ~1,005 lines

### Specialized Data Modules
| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `cerebro/data/movies.py` | 389 | MovieDataModule (contrastive pairs) | **ADOPT** |
| `cerebro/data/labram_pretrain.py` | 319 | LaBraMPretrainDataModule (masked tokens) | **ADOPT** |
| `cerebro/data/jepa_pretrain.py` | 450 | JEPAPretrainDataModule (JEPA pretraining) | **ADOPT** |

**Total specialized data modules**: ~1,158 lines

### Loss Functions
| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `cerebro/losses/__init__.py` | 222 | InfoNCE, NTXentLoss, etc. | **ADOPT** |

### Utilities
| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `cerebro/scripts/run_autopsy.py` | 214 | Standalone autopsy script | **ADOPT** |

### New-Style Configs
| File | Purpose | Status |
|------|---------|--------|
| `configs/supervised_eegnex_challenge1.yaml` | EEGNeX Challenge 1 (new architecture) | **ADOPT** |
| `configs/supervised_eegnex_challenge1_mini.yaml` | EEGNeX Challenge 1 mini | **ADOPT** |
| `configs/supervised_eegnex_challenge1_submission.yaml` | EEGNeX Challenge 1 submission | **ADOPT** |
| `configs/supervised_jepa_challenge1.yaml` | SignalJEPA Challenge 1 | **ADOPT** |
| `configs/supervised_jepa_challenge1_mini.yaml` | SignalJEPA Challenge 1 mini | **ADOPT** |
| `configs/contrastive_eegnex_movies.yaml` | EEGNeX movie contrastive | **ADOPT** |
| `configs/contrastive_eegnex_movies_mini.yaml` | EEGNeX movie contrastive mini | **ADOPT** |
| `configs/jepa_phase1_pretrain.yaml` | JEPA Phase 1 pretraining | **ADOPT** |
| `configs/jepa_phase1_pretrain_mini.yaml` | JEPA Phase 1 pretraining mini | **ADOPT** |

---

## Files MODIFIED Between Branches (NEED MERGE)

### Critical Merges Required
| File | Current | Movie | Changes | Strategy |
|------|---------|-------|---------|----------|
| `cerebro/data/hbn.py` | 1061 lines | 177 lines | -1126, +142 | **KEEP current + ADD movie's specialized modules separately** |
| `cerebro/cli/train.py` | - | - | 239 changes | **REVIEW diffs, adopt movie improvements** |
| `cerebro/models/__init__.py` | - | - | +47 lines | **MERGE exports** |
| `configs/labram/finetune.yaml` | - | - | 122 changes | **REVIEW diffs, update for new architecture** |
| `configs/labram/pretrain.yaml` | - | - | 112 changes | **REVIEW diffs, update for new architecture** |

### Minor Merges
| File | Changes | Strategy |
|------|---------|----------|
| `cerebro/utils/contrastive_dataset.py` | 12 lines | **REVIEW and merge** |
| `cerebro/utils/movie_windows.py` | 103 lines | **REVIEW and merge** |
| `cerebro/models/labram/tokenizer.py` | -4 lines | **REVIEW diff** |

---

## Migration Plan

### Phase 1: Preserve Current Branch Functionality
1. ✅ Create worktree for movie branch (done)
2. ✅ Document all differences (done)
3. Create backup branch of current state: `git branch backup/pre-movie-merge`

### Phase 2: Port Architecture Components (No Data Loss)
1. **COPY** (not move) from movie branch:
   - `cerebro/models/architectures.py`
   - `cerebro/models/builders.py`
   - `cerebro/models/components/`
   - `cerebro/trainers/`
   - `cerebro/losses/`

2. **KEEP existing** from current branch:
   - `cerebro/data/tuh_edf.py`
   - `cerebro/data/tuh.py`
   - `cerebro/data/hbn.py` (current 1061-line version)

3. **ADD new** specialized data modules alongside existing:
   - `cerebro/data/movies.py`
   - `cerebro/data/labram_pretrain.py`
   - `cerebro/data/jepa_pretrain.py`
   - `cerebro/data/challenge1.py` (extract from current hbn.py)
   - `cerebro/data/challenge2.py` (extract from current hbn.py)

### Phase 3: Update Configs
1. **KEEP** TUH configs from current branch
2. **ADD** new-style configs from movie branch
3. **MIGRATE** old-style Challenge configs to new architecture

### Phase 4: Merge Modified Files
1. **cerebro/cli/train.py**: Review 239-line diff, adopt improvements
2. **cerebro/models/__init__.py**: Merge exports
3. **configs/labram/*.yaml**: Update for new architecture

### Phase 5: Testing
1. Test TUH data modules still work
2. Test HBNDataModule backward compatibility
3. Test new architecture with Challenge 1 baselines
4. Test specialized data modules (movies, labram_pretrain)

---

## Key Decisions

### Decision 1: HBN Data Module Strategy
**Choice**: Keep current branch's full HBNDataModule (1061 lines) AND add movie branch's specialized modules

**Rationale**:
- Current HBNDataModule has comprehensive logic for all modes
- Movie branch's dispatcher pattern is elegant but loses functionality
- Best of both: full module for flexibility + specialized modules for clarity

**Implementation**:
```python
# Current branch (KEEP)
cerebro/data/hbn.py  # Full HBNDataModule with data_req parameter

# Add from movie branch
cerebro/data/challenge1.py  # Extract Challenge1 logic from HBNDataModule
cerebro/data/challenge2.py  # Extract Challenge2 logic from HBNDataModule
cerebro/data/movies.py      # Movie contrastive pairs
```

### Decision 2: TUH Data Modules
**Choice**: Keep ALL TUH modules from current branch

**Rationale**:
- Recent work on Zarr caching, interrupt handling
- Movie branch removed TUH support entirely
- Multi-phase plan requires TUH for Phase 1

**Files to preserve**:
- `cerebro/data/tuh_edf.py` (1327 lines) - Primary TUH implementation
- `cerebro/data/tuh.py` (557 lines) - HDF5 fallback
- All TUH configs and documentation

### Decision 3: Architecture Components
**Choice**: Adopt ALL architecture components from movie branch

**Rationale**:
- Production-ready quality
- Exactly what we need for multi-phase training
- No equivalent in current branch

**Components to adopt**:
- Registry pattern (`builders.py`)
- Compositional models (`architectures.py`)
- Encoder/decoder components (`components/`)
- Trainer modules (`trainers/`)
- Loss functions (`losses/`)

---

## Verification Checklist

Before completing migration:
- [ ] TUH data modules load correctly
- [ ] TUH configs work
- [ ] HBNDataModule backward compatibility (old configs still work)
- [ ] New architecture configs work (supervised, contrastive)
- [ ] LaBraM pretrain/finetune still works
- [ ] Challenge 1 baseline runs with new architecture
- [ ] All tests pass
- [ ] No import errors
- [ ] Documentation updated

---

## File Count Summary

**Current branch unique files**: 15 files (~3,852 lines, mostly TUH)
**Movie branch unique files**: 26 files (~5,974 lines, architecture + trainers)
**Modified files to merge**: 8 files
**Total integration effort**: ~34 files to review/merge

---

**Last Updated**: 2025-10-27 19:45
**Author**: Claude (with Varun's requirements)
