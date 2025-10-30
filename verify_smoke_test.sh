#!/bin/bash
# Verify smoke test pipeline completion

echo "=== SMOKE TEST VERIFICATION ==="
echo ""

PID=839278
if ps -p $PID > /dev/null 2>&1; then
    echo "Status: RUNNING (PID $PID)"
    echo ""
    echo "Latest progress:"
    tail -50 /tmp/smoke_test_complete.log | grep -E "^\[|✓|Epoch.*100%" | tail -5
else
    echo "Status: COMPLETED"
    echo ""
    echo "=== Final Results ==="
    tail -150 /tmp/smoke_test_complete.log | grep -E "^\[|✓|stopped" | tail -15
    echo ""

    # Check for all 7 stages
    echo "=== Stage Completion Check ==="
    grep -c "✓ Checkpoint:" /tmp/smoke_test_complete.log || echo "0"
    echo "/7 stages completed"
    echo ""

    # Check for errors
    if grep -q "Error\|Traceback" /tmp/smoke_test_complete.log; then
        echo "⚠️  ERRORS DETECTED"
        tail -100 /tmp/smoke_test_complete.log | grep -A 5 "Error\|Traceback" | tail -20
    else
        echo "✅ NO ERRORS"
    fi
fi
