#!/bin/bash

################################################################################
# SSL & Fine-Tuning Experiment Runner - With Error Recovery
# Runs experiments sequentially, skips failed ones, and continues
################################################################################

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$PROJECT_ROOT/logs/batch_runs"
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
LOG_FILE="$LOG_DIR/batch_${TIMESTAMP}.log"
RESULTS_FILE="$LOG_DIR/results_${TIMESTAMP}.txt"

# Create log directory
mkdir -p "$LOG_DIR"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ ERROR: $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ SUCCESS: $1" | tee -a "$LOG_FILE"
}

log_skip() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⏭️  SKIPPED: $1" | tee -a "$LOG_FILE"
}

log "=========================================="
log "Starting SSL & Fine-Tuning Experiments"
log "=========================================="
log "Timestamp: $TIMESTAMP"
log "Log file: $LOG_FILE"
log "Results file: $RESULTS_FILE"
log ""

# Array of experiments
experiments=(
    "baselines/baseline_resnet18_sample50"
    "baselines/baseline_resnet18"
    "baselines/baseline_resnet50"
    "transfer_learning/imagenet_resnet18"
    "transfer_learning/imagenet_resnet18_sample50"
    "transfer_learning/imagenet_resnet50"
    "ssl_pretraining/simclr_resnet18"
    "ssl_pretraining/simclr_resnet50"
    "linear_probing/simclr_resnet18_lp"
    "linear_probing/simclr_resnet18_lp_sample50"
    "linear_probing/simclr_resnet50_lp"
    "full_finetuning/simclr_resnet18_ft"
    "full_finetuning/simclr_resnet18_ft_sample50"
    "full_finetuning/simclr_resnet50_ft"
)

# Tracking variables
total=${#experiments[@]}
current=1
successful=0
failed=0

# Initialize results file
{
    echo "Experiment Results - $TIMESTAMP"
    echo "========================================"
    echo ""
} > "$RESULTS_FILE"

log "Total experiments to run: $total"
log ""

# Run each experiment
for exp in "${experiments[@]}"; do
    log "=========================================="
    log "[$current/$total] Running: $exp"
    log "=========================================="

    # Create experiment-specific log
    exp_name=$(echo "$exp" | tr '/' '_')
    exp_log="$LOG_DIR/exp_${current}_${exp_name}_${TIMESTAMP}.log"

    # Run experiment with timeout (48 hours max per experiment)
    timeout 172800 python experiments/run_experiment.py --experiment "$exp" > "$exp_log" 2>&1
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        log_success "Completed: $exp"
        echo "✅ $exp - SUCCESS" >> "$RESULTS_FILE"
        successful=$((successful + 1))
    else
        # Check if timeout occurred
        if [ $exit_code -eq 124 ]; then
            log_error "Timeout (48h) - $exp"
            echo "⏱️  $exp - TIMEOUT" >> "$RESULTS_FILE"
        else
            log_error "Failed with exit code $exit_code - $exp"
            echo "❌ $exp - FAILED (exit code: $exit_code)" >> "$RESULTS_FILE"
        fi

        failed=$((failed + 1))

        # Show last 20 lines of error
        log "Last 20 lines of error log:"
        tail -20 "$exp_log" | tee -a "$LOG_FILE"
        log ""

        # Continue to next experiment instead of exiting
        log_skip "Continuing with next experiment..."
    fi

    log "Experiment log: $exp_log"
    log ""

    current=$((current + 1))

    # Optional: Add delay between experiments (for GPU memory cleanup)
    if [ $current -le $total ]; then
        log "Waiting 60 seconds before next experiment (GPU memory cleanup)..."
        sleep 60
    fi
done

# Generate summary
log "=========================================="
log "BATCH RUN SUMMARY"
log "=========================================="

success_rate=$(awk "BEGIN {printf \"%.2f\", ($successful/$total)*100}")

{
    echo ""
    echo "=========================================="
    echo "BATCH RUN SUMMARY"
    echo "=========================================="
    echo "Timestamp: $TIMESTAMP"
    echo ""
    echo "Results:"
    echo "--------"
    echo "Total experiments: $total"
    echo "Successful: $successful ✅"
    echo "Failed: $failed ❌"
    echo "Success rate: ${success_rate}%"
    echo ""

    if [ $failed -gt 0 ]; then
        echo "Failed experiments:"
        grep "❌" "$RESULTS_FILE" | sed 's/^/  /'
        echo ""
    fi

    echo "Log files:"
    echo "  Main log: $LOG_FILE"
    echo "  Results: $RESULTS_FILE"
    echo "  Experiment logs: $LOG_DIR/exp_*"
    echo ""
} | tee -a "$LOG_FILE" >> "$RESULTS_FILE"

log ""
log "=========================================="
log "Batch run completed!"
log "=========================================="
log "Results saved to: $RESULTS_FILE"
log ""

# Exit with appropriate code
if [ $failed -eq 0 ]; then
    log_success "All experiments completed successfully!"
    exit 0
else
    log_error "$failed experiment(s) failed, $successful succeeded"
    exit 1
fi