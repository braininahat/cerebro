#!/bin/bash
set -e

echo "========================================="
echo "SMOKE TEST PIPELINE (7 stages, 20 steps each)"
echo "========================================="
echo ""

echo "[1/7] LaBraM Codebook (20 steps)..."
WANDB_MODE=offline uv run cerebro fit --config configs/labram/codebook_smoke.yaml
echo "✓ Checkpoint: outputs/labram/codebook_smoke/checkpoints/last.ckpt"
echo ""

echo "[2/7] LaBraM MEM Pretrain (20 steps)..."
WANDB_MODE=offline uv run cerebro fit --config configs/labram/pretrain_smoke.yaml
echo "✓ Checkpoint: outputs/labram/pretrain_smoke/checkpoints/last.ckpt"
echo ""

echo "[3/7] LaBraM Challenge 1 (20 steps)..."
WANDB_MODE=offline uv run cerebro fit --config configs/labram/finetune_challenge1_smoke.yaml
echo "✓ Checkpoint: outputs/labram/finetune_c1_smoke/checkpoints/last.ckpt"
echo ""

echo "[4/7] LaBraM Challenge 2 (20 steps)..."
WANDB_MODE=offline uv run cerebro fit --config configs/labram/finetune_challenge2_smoke.yaml
echo "✓ Checkpoint: outputs/labram/finetune_c2_smoke/checkpoints/last.ckpt"
echo ""

echo "[5/7] SignalJEPA Pretrain (20 steps)..."
WANDB_MODE=offline uv run cerebro fit --config configs/jepa_pretrain_smoke.yaml
echo "✓ Checkpoint: outputs/jepa/pretrain_smoke/checkpoints/last.ckpt"
echo ""

echo "[6/7] SignalJEPA Challenge 1 (20 steps)..."
WANDB_MODE=offline uv run cerebro fit --config configs/jepa_c1_smoke.yaml
echo "✓ Checkpoint: outputs/jepa/finetune_c1_smoke/checkpoints/last.ckpt"
echo ""

echo "[7/7] SignalJEPA Challenge 2 (20 steps)..."
WANDB_MODE=offline uv run cerebro fit --config configs/jepa_c2_smoke.yaml
echo "✓ Checkpoint: outputs/jepa/finetune_c2_smoke/checkpoints/last.ckpt"
echo ""

echo "========================================="
echo "✅ ALL 7 STAGES COMPLETE"
echo "========================================="
