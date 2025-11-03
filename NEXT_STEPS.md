# IMMEDIATE NEXT STEPS

## Run Complete Smoke Test

The smoke test pipeline script needs to be created and run. Here's what to do:

1. **Create the smoke test script**:
```bash
cat > run_smoke_test_pipeline.sh << 'SCRIPT'
#!/bin/bash
set -e

echo "[1/7] LaBraM Codebook (20 steps)..."
WANDB_MODE=offline uv run cerebro fit --config configs/labram/codebook_smoke.yaml
echo "✓ Checkpoint: outputs/labram/codebook_smoke/checkpoints/last.ckpt"

echo "[2/7] LaBraM MEM Pretrain (20 steps)..."
WANDB_MODE=offline uv run cerebro fit --config configs/labram/pretrain_smoke.yaml
echo "✓ Checkpoint: outputs/labram/pretrain_smoke/checkpoints/last.ckpt"

echo "[3/7] LaBraM Challenge 1 (20 steps)..."
WANDB_MODE=offline uv run cerebro fit --config configs/labram/finetune_challenge1_smoke.yaml
echo "✓ Checkpoint: outputs/labram/finetune_c1_smoke/checkpoints/last.ckpt"

echo "[4/7] LaBraM Challenge 2 (20 steps)..."
WANDB_MODE=offline uv run cerebro fit --config configs/labram/finetune_challenge2_smoke.yaml
echo "✓ Checkpoint: outputs/labram/finetune_c2_smoke/checkpoints/last.ckpt"

echo "[5/7] SignalJEPA Pretrain (20 steps)..."
WANDB_MODE=offline uv run cerebro fit --config configs/jepa_pretrain_smoke.yaml
echo "✓ Checkpoint: outputs/jepa/pretrain_smoke/checkpoints/last.ckpt"

echo "[6/7] SignalJEPA Challenge 1 (20 steps)..."
WANDB_MODE=offline uv run cerebro fit --config configs/jepa_c1_smoke.yaml
echo "✓ Checkpoint: outputs/jepa/finetune_c1_smoke/checkpoints/last.ckpt"

echo "[7/7] SignalJEPA Challenge 2 (20 steps)..."
WANDB_MODE=offline uv run cerebro fit --config configs/jepa_c2_smoke.yaml
echo "✓ Checkpoint: outputs/jepa/finetune_c2_smoke/checkpoints/last.ckpt"

echo ""
echo "========================================="
echo "✅ ALL 7 STAGES COMPLETE"
echo "========================================="
SCRIPT

chmod +x run_smoke_test_pipeline.sh
```

2. **Run it**:
```bash
./run_smoke_test_pipeline.sh 2>&1 | tee smoke_test_output.log
```

3. **Verify completion**:
```bash
grep "✓ Checkpoint:" smoke_test_output.log | wc -l  # Should show 7
grep "ALL 7 STAGES COMPLETE" smoke_test_output.log  # Should show success message
```

## Expected Runtime

- Total: ~15-20 minutes
- Per stage: ~2-3 minutes (20 steps each)

## Success Criteria

- All 7 checkpoints created
- No errors in output
- No NaN losses in Challenge 2
- SignalJEPA successfully loads pretrained weights

## After Smoke Test Passes

Then run the remaining verification steps from the original plan:

1. Test EEGNeX supervised baselines
2. Test SignalJEPA supervised baselines  
3. Test submission.zip creation
4. Test local scoring
5. Create performance table

All infrastructure fixes are in place - the smoke test just validates everything works end-to-end!
