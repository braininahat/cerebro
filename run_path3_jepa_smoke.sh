#!/bin/bash
set -e

echo "========================================="
echo "PATH 3 SMOKE TEST: SignalJEPA (4 stages)"
echo "Pretrain → Contrastive → Supervised"
echo "========================================="
echo ""

echo "[1/4] Phase 1: JEPA Pretrain (20 steps)..."
WANDB_MODE=offline uv run cerebro fit --config configs/jepa_pretrain_smoke.yaml
echo "✓ Checkpoint: outputs/jepa/pretrain_smoke/checkpoints/last.ckpt"
echo ""

echo "[2/4] Phase 2: Movie Contrastive Learning (20 steps)..."
WANDB_MODE=offline uv run cerebro fit --config configs/jepa_phase2_contrastive_smoke.yaml
echo "✓ Checkpoint: outputs/jepa/phase2_contrastive_smoke/checkpoints/last.ckpt"
echo ""

echo "[3/4] Phase 3: Challenge 1 Supervised Finetuning (20 steps)..."
WANDB_MODE=offline uv run cerebro fit --config configs/jepa_phase3_challenge1_smoke.yaml
echo "✓ Checkpoint: outputs/jepa/phase3_c1_smoke/checkpoints/last.ckpt"
echo ""

echo "[4/4] Phase 3: Challenge 2 Supervised Finetuning (20 steps)..."
WANDB_MODE=offline uv run cerebro fit --config configs/jepa_phase3_challenge2_smoke.yaml
echo "✓ Checkpoint: outputs/jepa/phase3_c2_smoke/checkpoints/last.ckpt"
echo ""

echo "========================================="
echo "✅ PATH 3 COMPLETE (SignalJEPA)"
echo "========================================="
echo ""
echo "Summary:"
echo "  Phase 1: Self-supervised pretraining (JEPA)"
echo "  Phase 2: Contrastive learning (movie ISC)"
echo "  Phase 3: Supervised finetuning (Challenge 1 & 2)"
echo ""
echo "Checkpoints:"
echo "  - outputs/jepa/pretrain_smoke/checkpoints/last.ckpt"
echo "  - outputs/jepa/phase2_contrastive_smoke/checkpoints/last.ckpt"
echo "  - outputs/jepa/phase3_c1_smoke/checkpoints/last.ckpt"
echo "  - outputs/jepa/phase3_c2_smoke/checkpoints/last.ckpt"
