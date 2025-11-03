#!/bin/bash
#
# Run complete foundation model pipeline end-to-end (mini versions, R1 only, 2 epochs each)
# Tests all 7 stages of LaBraM + SignalJEPA training
#
# Total estimated time: ~2-3 hours
#

set -e  # Exit on error

echo "========================================"
echo " FOUNDATION MODEL PIPELINE (MINI)"
echo "========================================"
echo ""
echo "This will run all 7 training stages:"
echo "  - LaBraM: Codebook → Pretrain → C1 → C2 (4 stages)"
echo "  - SignalJEPA: Pretrain → C1 → C2 (3 stages)"
echo ""
echo "Each stage: R1 only, 2 epochs"
echo "Estimated time: 2-3 hours total"
echo ""

# ============================================================================
# LaBraM Pipeline
# ============================================================================

echo ""
echo "=========================================="
echo " LAB RAM PIPELINE"
echo "=========================================="

# Phase 1: Codebook Training
echo ""
echo "[1/7] LaBraM Codebook Training..."
WANDB_MODE=offline uv run cerebro fit --config configs/labram/codebook_mini.yaml
echo "✓ Codebook training complete"
echo "  Checkpoint: outputs/labram/codebook_mini/checkpoints/tokenizer-last.ckpt"

# Phase 2: MEM Pretraining
echo ""
echo "[2/7] LaBraM MEM Pretraining..."
WANDB_MODE=offline uv run cerebro fit --config configs/labram/pretrain_mini.yaml
echo "✓ MEM pretraining complete"
echo "  Checkpoint: outputs/labram/pretrain_mini/checkpoints/pretrain-last.ckpt"

# Phase 3a: Challenge 1 Finetuning
echo ""
echo "[3/7] LaBraM Challenge 1 Finetuning..."
WANDB_MODE=offline uv run cerebro fit --config configs/labram/finetune_challenge1_mini.yaml
echo "✓ Challenge 1 finetuning complete"
echo "  Checkpoint: outputs/labram/finetune_c1_mini/checkpoints/challenge1-last.ckpt"

# Phase 3b: Challenge 2 Finetuning
echo ""
echo "[4/7] LaBraM Challenge 2 Finetuning..."
WANDB_MODE=offline uv run cerebro fit --config configs/labram/finetune_challenge2_mini.yaml
echo "✓ Challenge 2 finetuning complete"
echo "  Checkpoint: outputs/labram/finetune_c2_mini/checkpoints/challenge2-last.ckpt"

# ============================================================================
# SignalJEPA Pipeline
# ============================================================================

echo ""
echo "=========================================="
echo " SIGNAL JEPA PIPELINE"
echo "=========================================="

# Phase 1: JEPA Pretraining
echo ""
echo "[5/7] SignalJEPA Pretraining..."
WANDB_MODE=offline uv run cerebro fit --config configs/jepa_pretrain_mini.yaml
echo "✓ JEPA pretraining complete"
echo "  Checkpoint: outputs/jepa/pretrain_mini/checkpoints/pretrain-last.ckpt"

# Phase 2a: Challenge 1 Finetuning
echo ""
echo "[6/7] SignalJEPA Challenge 1 Finetuning..."
WANDB_MODE=offline uv run cerebro fit --config configs/jepa_c1_mini.yaml
echo "✓ Challenge 1 finetuning complete"
echo "  Checkpoint: outputs/jepa/finetune_c1_mini/checkpoints/challenge1-last.ckpt"

# Phase 2b: Challenge 2 Finetuning
echo ""
echo "[7/7] SignalJEPA Challenge 2 Finetuning..."
WANDB_MODE=offline uv run cerebro fit --config configs/jepa_c2_mini.yaml
echo "✓ Challenge 2 finetuning complete"
echo "  Checkpoint: outputs/jepa/finetune_c2_mini/checkpoints/challenge2-last.ckpt"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "=========================================="
echo " PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "All 7 stages completed successfully:"
echo ""
echo "LaBraM:"
echo "  ✓ Codebook training"
echo "  ✓ MEM pretraining"
echo "  ✓ Challenge 1 finetuning"
echo "  ✓ Challenge 2 finetuning"
echo ""
echo "SignalJEPA:"
echo "  ✓ JEPA pretraining"
echo "  ✓ Challenge 1 finetuning"
echo "  ✓ Challenge 2 finetuning"
echo ""
echo "Checkpoints saved in outputs/labram/ and outputs/jepa/"
echo "Logs saved in corresponding directories"
echo ""
echo "Next steps:"
echo "  1. Review training metrics in wandb logs"
echo "  2. Compare pretrained vs supervised-only performance"
echo "  3. Run full training with complete dataset (R1-R11)"
echo ""
