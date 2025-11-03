#!/bin/bash
# Comprehensive Verification Test Suite
# Runs all smoke tests for supervised baselines + submission + scoring

set -e

echo "==========================================="
echo "COMPREHENSIVE VERIFICATION TEST SUITE"
echo "==========================================="
echo ""

# Function to check if a checkpoint exists
check_checkpoint() {
    local ckpt=$1
    if [ -f "$ckpt" ]; then
        echo "  ✓ Checkpoint exists: $ckpt"
        return 0
    else
        echo "  ✗ Checkpoint missing: $ckpt"
        return 1
    fi
}

# Part 1: Verify main pipeline checkpoints
echo "=== PART 1: Verify Main Pipeline Checkpoints ==="
echo ""
check_checkpoint "outputs/labram/codebook_smoke/checkpoints/last.ckpt"
check_checkpoint "outputs/labram/pretrain_smoke/checkpoints/last.ckpt"
check_checkpoint "outputs/labram/finetune_c1_smoke/checkpoints/last.ckpt"
check_checkpoint "outputs/labram/finetune_c2_smoke/checkpoints/last.ckpt"
check_checkpoint "outputs/jepa/pretrain_smoke/checkpoints/last.ckpt"
check_checkpoint "outputs/jepa/finetune_c1_smoke/checkpoints/last.ckpt"
check_checkpoint "outputs/jepa/finetune_c2_smoke/checkpoints/last.ckpt"
echo ""

# Part 2: Test EEGNeX supervised baselines
echo "=== PART 2: EEGNeX Supervised Baselines (20 steps each) ==="
echo ""

echo "[1/4] EEGNeX Challenge 1..."
WANDB_MODE=offline uv run cerebro fit --config configs/supervised_eegnex_challenge1_smoke.yaml
check_checkpoint "outputs/supervised_eegnex_c1_smoke/checkpoints/last.ckpt"
echo ""

echo "[2/4] EEGNeX Challenge 2..."
WANDB_MODE=offline uv run cerebro fit --config configs/supervised_eegnex_challenge2_smoke.yaml
check_checkpoint "outputs/supervised_eegnex_c2_smoke/checkpoints/last.ckpt"
echo ""

echo "[3/4] SignalJEPA Supervised Challenge 1..."
WANDB_MODE=offline uv run cerebro fit --config configs/supervised_jepa_challenge1_smoke.yaml
check_checkpoint "outputs/supervised_jepa_c1_smoke/checkpoints/last.ckpt"
echo ""

echo "[4/4] SignalJEPA Supervised Challenge 2..."
WANDB_MODE=offline uv run cerebro fit --config configs/supervised_jepa_challenge2_smoke.yaml
check_checkpoint "outputs/supervised_jepa_c2_smoke/checkpoints/last.ckpt"
echo ""

# Part 3: Test submission creation
echo "=== PART 3: Submission Creation ==="
echo ""

echo "Creating test submission from EEGNeX baselines..."
if uv run cerebro build-submission \
    --challenge1-ckpt outputs/supervised_eegnex_c1_smoke/checkpoints/last.ckpt \
    --challenge2-ckpt outputs/supervised_eegnex_c2_smoke/checkpoints/last.ckpt \
    -o test_submission.zip; then
    echo "✓ Submission created: test_submission.zip"
    ls -lh test_submission.zip
else
    echo "✗ Submission creation failed"
    exit 1
fi
echo ""

# Part 4: Test local scoring
echo "=== PART 4: Local Scoring ==="
echo ""

if [ -f "test_submission.zip" ]; then
    echo "Running local scoring on test submission..."
    if uv run cerebro-score test_submission.zip --fast-dev-run; then
        echo "✓ Local scoring completed successfully"
    else
        echo "✗ Local scoring failed"
        exit 1
    fi
else
    echo "✗ test_submission.zip not found, skipping scoring"
fi
echo ""

# Summary
echo "==========================================="
echo "✅ ALL VERIFICATION TESTS COMPLETE"
echo "==========================================="
echo ""
echo "Summary of working pipelines:"
echo "  • LaBraM: Codebook → Pretrain → Challenge 1/2 Finetuning"
echo "  • SignalJEPA: Pretrain → Challenge 1/2 Finetuning"
echo "  • EEGNeX: Supervised Challenge 1/2 (from scratch)"
echo "  • SignalJEPA: Supervised Challenge 1/2 (from scratch)"
echo "  • Submission creation: ✓"
echo "  • Local scoring: ✓"
echo ""
echo "Ready for production runs!"
