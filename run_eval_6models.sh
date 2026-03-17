#!/bin/bash
# ============================================================================
# run_eval_6models.sh - 6个 chatbot SFT 模型并行测评
# 2 GPUs × 3 concurrent models per GPU = 6 parallel evaluations
# 实时进度看板 + 自动跳过已完成的指标组
#
# Usage:
#   bash run_eval_6models.sh               # 全量 (fast + slow)
#   bash run_eval_6models.sh fast           # 仅跑 loglikelihood 指标
#   bash run_eval_6models.sh generative     # 仅跑 generate_until 指标
#   bash run_eval_6models.sh --force        # 强制重新评测
# ============================================================================

# NOTE: Do NOT use 'set -e' here. Bash arithmetic like ((x++)) returns 1
# when x was 0, and 'set -e' treats that as a fatal error.
set -uo pipefail

# ─── Configuration ──────────────────────────────────────────────────────────
NUM_GPUS=${NUM_GPUS:-2}
JOBS_PER_GPU=${JOBS_PER_GPU:-3}
BSZ=${BSZ:-8}
FORCE=0
LOGDIR="eval_logs/chatbot"

# NOTE: Do NOT use $GROUPS — it is a reserved bash readonly array variable.
EVAL_GROUP="all"
CUSTOM_TASKS=""

# Parse args
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        --tasks=*) CUSTOM_TASKS="${arg#--tasks=}" ;;
        all|fast|generative|base_retain|chatbot|advanced|new) EVAL_GROUP="$arg" ;;
        *) echo "[ERROR] Unknown argument: $arg (use: all, fast, generative, base_retain, chatbot, advanced, new, --force, --tasks=task1,task2)"; exit 1 ;;
    esac
done

# 6 models to evaluate
MODELS=(
    "out/L28-D3584-qwerky7_qwen2-6_chatbot_slimorca/rwkv-step1500-197M.pth"
    "out/L28-D3584-qwerky7_qwen2-6_chatbot_slimorca/rwkv-step150-20M.pth"
    "out/L28-D3584-qwerky7_qwen2-6_chatbot_openhermes/rwkv-step900-118M.pth"
    "out/L28-D3584-qwerky7_qwen2-6_chatbot_ultrachat/rwkv-step1500-197M.pth"
    "out/L28-D3584-qwerky7_qwen2-6_chatbot_ultrachat/rwkv-step150-20M.pth"
    "out/L28-D3584-qwerky7_qwen2-4_BOA/rwkv-1.pth"
)

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║          RADLADS 6-Model Parallel Benchmark Evaluation            ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
if [ -n "$CUSTOM_TASKS" ]; then
    echo "  Mode:          custom (--tasks=$CUSTOM_TASKS)"
else
    echo "  Mode:          $EVAL_GROUP"
fi
echo "  GPUs:          $NUM_GPUS (GPU 0, GPU 1)"
echo "  Jobs per GPU:  $JOBS_PER_GPU"
echo "  Max parallel:  $((NUM_GPUS * JOBS_PER_GPU))"
echo "  Batch size:    $BSZ"
echo "  Force re-eval: $( [ $FORCE -eq 1 ] && echo 'YES' || echo 'no' )"
echo "  Models:        ${#MODELS[@]}"
echo ""

# Verify all models exist
MISSING=0
for m in "${MODELS[@]}"; do
    if [ ! -f "$m" ]; then
        echo "  [MISSING] $m"
        MISSING=$((MISSING + 1))
    else
        echo "  [OK] $m"
    fi
done
echo ""

if [ $MISSING -gt 0 ]; then
    echo "[ERROR] $MISSING model(s) not found. Aborting."
    exit 1
fi

mkdir -p "$LOGDIR"

# ─── Job tracking arrays ───────────────────────────────────────────────────
declare -A JOB_PIDS       # key -> PID
declare -A JOB_GPUS       # key -> GPU id
declare -A JOB_LOGS       # key -> logfile path
declare -A JOB_LABELS     # key -> display label
declare -A JOB_STATUS     # key -> running|done|fail
declare -A GPU_RUNNING    # gpu_id -> count of running jobs

for ((g=0; g<NUM_GPUS; g++)); do
    GPU_RUNNING[$g]=0
done

TOTAL=${#MODELS[@]}
LAUNCHED=0
COMPLETED=0
FAILED_COUNT=0
NEXT_IDX=0

# ─── Progress parsing ──────────────────────────────────────────────────────
get_progress() {
    local logfile="$1"
    if [ ! -f "$logfile" ]; then
        echo "-1"
        return 0
    fi
    local tail_text
    tail_text=$(tail -c 4096 "$logfile" 2>/dev/null || true)

    # Match lm_eval progress: "Running loglikelihood requests ... 45%|..."
    # or eval_chatbot patterns like "[OK]" or "Running"
    local pct
    pct=$(echo "$tail_text" | grep -oE '[0-9]+%' | tail -1 | grep -oE '[0-9]+' || true)
    if [ -n "$pct" ]; then
        echo "$pct"
        return 0
    fi

    # Check for setup phase
    if echo "$tail_text" | grep -qE 'Loading model|RWKV_MODEL_TYPE|Overwriting default' 2>/dev/null; then
        echo "0"
        return 0
    fi

    # Check for skip/completion
    if echo "$tail_text" | grep -qE '\[SKIP\]|\[OK\]' 2>/dev/null; then
        echo "50"
        return 0
    fi

    echo "-1"
    return 0
}

# ─── Build progress bar string ──────────────────────────────────────────────
make_bar() {
    local pct=$1
    local width=25
    local filled=$((width * pct / 100))
    local empty=$((width - filled))
    local bar=""
    local i

    for ((i=0; i<filled; i++)); do
        bar+="="
    done
    for ((i=0; i<empty; i++)); do
        bar+="-"
    done
    echo "$bar"
}

# ─── Dashboard rendering ───────────────────────────────────────────────────
DASHBOARD_LINES=0

draw_dashboard() {
    # Clear previous dashboard
    if [ $DASHBOARD_LINES -gt 0 ]; then
        printf '\033[%dA\033[J' "$DASHBOARD_LINES"
    fi

    local lines=1  # Start at 1 to avoid ((0++)) == false issue

    # Header
    echo "-------------------------------------------------------------------------"
    lines=$((lines + 1))

    printf "  %-35s %-6s %s\n" "Model" "GPU" "Progress"
    lines=$((lines + 1))

    echo "-------------------------------------------------------------------------"
    lines=$((lines + 1))

    # All jobs sorted by key
    local sorted_keys
    sorted_keys=$(echo "${!JOB_STATUS[@]}" | tr ' ' '\n' | sort)

    for key in $sorted_keys; do
        local status="${JOB_STATUS[$key]}"
        local gpu="${JOB_GPUS[$key]}"
        local label="${JOB_LABELS[$key]}"
        local logfile="${JOB_LOGS[$key]}"

        # Truncate label
        if [ ${#label} -gt 35 ]; then
            label="${label:0:32}..."
        fi

        if [ "$status" = "running" ]; then
            local pct
            pct=$(get_progress "$logfile")

            if [ -z "$pct" ] || [ "$pct" = "-1" ]; then
                printf "  %-35s GPU%-3s > [...........................] ...\n" "$label" "$gpu"
            else
                local bar
                bar=$(make_bar "$pct")
                printf "  %-35s GPU%-3s > [%s] %3d%%\n" "$label" "$gpu" "$bar" "$pct"
            fi
            lines=$((lines + 1))
        elif [ "$status" = "done" ]; then
            printf "  %-35s GPU%-3s   [=========================] done\n" "$label" "$gpu"
            lines=$((lines + 1))
        elif [ "$status" = "fail" ]; then
            printf "  %-35s GPU%-3s   FAILED\n" "$label" "$gpu"
            lines=$((lines + 1))
        fi
    done

    # Summary line
    echo "-------------------------------------------------------------------------"
    lines=$((lines + 1))

    local running=$((LAUNCHED - COMPLETED - FAILED_COUNT))
    local fail_str=""
    if [ $FAILED_COUNT -gt 0 ]; then
        fail_str=" ($FAILED_COUNT failed)"
    fi

    local now
    now=$(date +%s)
    local elapsed=$((now - START_TIME))
    local elapsed_min=$((elapsed / 60))
    local elapsed_sec=$((elapsed % 60))

    printf "  Done: %d/%d%s | Running: %d | Time: %dm%02ds\n" \
        "$((COMPLETED + FAILED_COUNT))" "$TOTAL" "$fail_str" "$running" \
        "$elapsed_min" "$elapsed_sec"
    lines=$((lines + 1))

    DASHBOARD_LINES=$lines
}

# ─── Launch a job ───────────────────────────────────────────────────────────
launch_job() {
    local idx=$1
    local model="${MODELS[$idx]}"
    local dirbase
    dirbase=$(basename "$(dirname "$model")")
    local name
    name=$(basename "$model" .pth)
    local key="${dirbase##*_}/${name}"
    local logfile="$LOGDIR/${dirbase}_${name}.log"

    # Pick GPU with fewest running jobs
    local best_gpu=0
    local min_jobs=${GPU_RUNNING[0]}
    for ((g=1; g<NUM_GPUS; g++)); do
        if [ "${GPU_RUNNING[$g]}" -lt "$min_jobs" ]; then
            min_jobs=${GPU_RUNNING[$g]}
            best_gpu=$g
        fi
    done

    # Launch eval subprocess
    # Do NOT pass --gpu: eval_chatbot.py would override CUDA_VISIBLE_DEVICES.
    # Let the shell-level CUDA_VISIBLE_DEVICES handle GPU assignment.
    local force_flag=""
    if [ $FORCE -eq 1 ]; then
        force_flag="--force"
    fi

    local tasks_flag=""
    if [ -n "$CUSTOM_TASKS" ]; then
        tasks_flag="--tasks $CUSTOM_TASKS"
    fi

    CUDA_VISIBLE_DEVICES=$best_gpu python eval_chatbot.py eval \
        --path "$model" \
        --bsz $BSZ \
        --group "$EVAL_GROUP" \
        $force_flag \
        $tasks_flag \
        > "$logfile" 2>&1 &

    local pid=$!

    JOB_PIDS[$key]=$pid
    JOB_GPUS[$key]=$best_gpu
    JOB_LOGS[$key]=$logfile
    JOB_LABELS[$key]=$key
    JOB_STATUS[$key]="running"
    GPU_RUNNING[$best_gpu]=$((GPU_RUNNING[$best_gpu] + 1))
    LAUNCHED=$((LAUNCHED + 1))
}

# ─── Check finished jobs ───────────────────────────────────────────────────
check_finished() {
    for key in "${!JOB_PIDS[@]}"; do
        if [ "${JOB_STATUS[$key]}" = "running" ]; then
            local pid=${JOB_PIDS[$key]}
            if ! kill -0 "$pid" 2>/dev/null; then
                # Process finished — get exit code
                local rc=0
                wait "$pid" 2>/dev/null || rc=$?
                local gpu=${JOB_GPUS[$key]}
                GPU_RUNNING[$gpu]=$((GPU_RUNNING[$gpu] - 1))

                if [ $rc -eq 0 ]; then
                    JOB_STATUS[$key]="done"
                    COMPLETED=$((COMPLETED + 1))
                else
                    JOB_STATUS[$key]="fail"
                    FAILED_COUNT=$((FAILED_COUNT + 1))
                fi
            fi
        fi
    done
}

# ─── Fill GPU slots ────────────────────────────────────────────────────────
fill_slots() {
    while [ $NEXT_IDX -lt $TOTAL ]; do
        local launched_any=0
        for ((g=0; g<NUM_GPUS; g++)); do
            if [ "${GPU_RUNNING[$g]}" -lt $JOBS_PER_GPU ] && [ $NEXT_IDX -lt $TOTAL ]; then
                launch_job $NEXT_IDX
                NEXT_IDX=$((NEXT_IDX + 1))
                launched_any=1
            fi
        done
        if [ $launched_any -eq 0 ]; then
            break
        fi
    done
}

# ─── Cleanup on Ctrl+C ─────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo ""
    echo "[INTERRUPTED] Killing all running eval processes..."
    for key in "${!JOB_PIDS[@]}"; do
        if [ "${JOB_STATUS[$key]}" = "running" ]; then
            kill "${JOB_PIDS[$key]}" 2>/dev/null || true
        fi
    done
    echo "Background processes killed. Logs saved to $LOGDIR/"
    exit 130
}
trap cleanup INT TERM

# ─── Main loop ──────────────────────────────────────────────────────────────
START_TIME=$(date +%s)

echo "Launching evaluations (2 GPUs × 3 models each)..."
echo ""

# Launch all 6 models immediately (3 per GPU)
fill_slots

# Poll until all done
while [ $((COMPLETED + FAILED_COUNT)) -lt $TOTAL ]; do
    sleep 3
    check_finished
    fill_slots
    draw_dashboard
done

# Final dashboard
draw_dashboard

# Final summary
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo "========================================================================="
echo "                        EVALUATION COMPLETE"
echo "========================================================================="
echo "  Total time:  ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "  Completed:   $((TOTAL - FAILED_COUNT)) / $TOTAL"
if [ $FAILED_COUNT -gt 0 ]; then
    echo "  Failed:      $FAILED_COUNT"
    echo ""
    echo "  Failed model logs:"
    for key in "${!JOB_STATUS[@]}"; do
        if [ "${JOB_STATUS[$key]}" = "fail" ]; then
            echo "    - ${JOB_LOGS[$key]}"
        fi
    done
fi
echo ""

# Generate final summary
echo "Generating summary..."
python eval_chatbot.py summary
echo ""
echo "Done! Results in eval_chatbot_results.json / eval_chatbot_summary.md"
