#!/bin/bash
# ============================================================================
# auto_eval.sh - 自动全量权重测评脚本
# 2 GPUs × 3 concurrent models per GPU = 6 parallel evaluations
# 实时进度条显示每个模型的测评进度
#
# Usage:
#   bash auto_eval.sh                        # eval ALL out/ dirs
#   bash auto_eval.sh out/some_dir           # eval one dir only
#   bash auto_eval.sh --force                # force re-eval completed ones
#   BSZ=4 bash auto_eval.sh                  # custom batch size
# ============================================================================

# NOTE: Do NOT use 'set -e' here. Bash arithmetic like ((x++)) returns 1
# when x was 0, and 'set -e' treats that as a fatal error.
set -uo pipefail

# ─── Configuration ──────────────────────────────────────────────────────────
NUM_GPUS=2
JOBS_PER_GPU=3
BSZ=${BSZ:-1}
FORCE=${FORCE:-0}
LOGDIR="eval_logs"
ALL_TASKS="lambada_openai,arc_easy,arc_challenge,hellaswag,winogrande,piqa,openbookqa,boolq"
COMMON_ARGS="-c configs/qwen7b.yaml -c configs/qwerky7.yaml --model.attention_type rwkv7_fla_chunk --model.ctx_len 4096 --precision bf16 --bsz $BSZ"

# Parse args
CKPT_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--force" ]; then
        FORCE=1
    else
        CKPT_ARGS+=("$arg")
    fi
done

# ─── Discover checkpoints ──────────────────────────────────────────────────
if [ ${#CKPT_ARGS[@]} -gt 0 ]; then
    CKPT_DIRS=("${CKPT_ARGS[@]}")
else
    CKPT_DIRS=(out/*)
fi

ALL_CKPTS=()
for dir in "${CKPT_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        for ckpt in "$dir"/rwkv-*.pth; do
            [ -f "$ckpt" ] && ALL_CKPTS+=("$ckpt")
        done
    fi
done

if [ ${#ALL_CKPTS[@]} -eq 0 ]; then
    echo "[ERROR] No checkpoints found under: ${CKPT_DIRS[*]}"
    exit 1
fi

mkdir -p "$LOGDIR"

# ─── Display plan ──────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                    RADLADS Auto Weight Evaluation                  ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Checkpoints found: ${#ALL_CKPTS[@]}"
echo "  GPUs:              $NUM_GPUS (GPU 0, GPU 1)"
echo "  Jobs per GPU:      $JOBS_PER_GPU"
echo "  Max parallel:      $((NUM_GPUS * JOBS_PER_GPU))"
echo "  Batch size:        $BSZ"
echo "  Tasks:             $ALL_TASKS"
echo "  Force re-eval:     $( [ $FORCE -eq 1 ] && echo 'YES' || echo 'no' )"
echo ""

for dir in "${CKPT_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        count=$(ls "$dir"/rwkv-*.pth 2>/dev/null | wc -l)
        echo "  $dir: $count checkpoint(s)"
    fi
done
echo ""
echo "========================================================================="

# ─── Skip already-evaluated checkpoints ─────────────────────────────────────
EVAL_CKPTS=()
SKIPPED=0
for ckpt in "${ALL_CKPTS[@]}"; do
    dirbase=$(basename "$(dirname "$ckpt")")
    name=$(basename "$ckpt" .pth)
    logfile="$LOGDIR/${dirbase}_${name}_all.log"

    if [ $FORCE -eq 0 ] && [ -f "$logfile" ] && grep -q "100%.*batches.*done" "$logfile" 2>/dev/null; then
        SKIPPED=$((SKIPPED + 1))
        echo "  [SKIP] $dirbase/$name (already completed)"
    else
        EVAL_CKPTS+=("$ckpt")
    fi
done

if [ $SKIPPED -gt 0 ]; then
    echo ""
    echo "  Skipped: $SKIPPED (already evaluated)"
fi

if [ ${#EVAL_CKPTS[@]} -eq 0 ]; then
    echo ""
    echo "  All checkpoints already evaluated! Use --force to re-evaluate."
    exit 0
fi

echo "  Remaining: ${#EVAL_CKPTS[@]} checkpoints to evaluate"
echo ""

# ─── Job tracking arrays ───────────────────────────────────────────────────
declare -A JOB_PIDS       # ckpt_key -> PID
declare -A JOB_GPUS       # ckpt_key -> GPU id
declare -A JOB_LOGS       # ckpt_key -> logfile path
declare -A JOB_LABELS     # ckpt_key -> display label
declare -A JOB_STATUS     # ckpt_key -> running|done|fail
declare -A GPU_RUNNING    # gpu_id -> count of running jobs

for ((g=0; g<NUM_GPUS; g++)); do
    GPU_RUNNING[$g]=0
done

TOTAL=${#EVAL_CKPTS[@]}
LAUNCHED=0
COMPLETED=0
FAILED=0
NEXT_IDX=0

# ─── Progress parsing ──────────────────────────────────────────────────────
get_progress() {
    local logfile="$1"
    if [ ! -f "$logfile" ]; then
        echo "-1"
        return 0
    fi
    # Read last 4KB of log to find progress bar output
    local tail_text
    tail_text=$(tail -c 4096 "$logfile" 2>/dev/null || true)

    # Match: 35% (120/340 batches) - use grep -o with extended regex
    local pct
    pct=$(echo "$tail_text" | grep -oE '[0-9]+% \([0-9]+/[0-9]+ batches\)' | tail -1 | grep -oE '^[0-9]+' || true)
    if [ -n "$pct" ]; then
        echo "$pct"
        return 0
    fi

    # Check for setup phase
    if echo "$tail_text" | grep -qE 'Loading model|RWKV_MODEL_TYPE|Overwriting default' 2>/dev/null; then
        echo "0"
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
    local line

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

    # Queued count
    local queued=$((TOTAL - LAUNCHED))
    if [ $queued -gt 0 ]; then
        printf "  ... %d more queued\n" "$queued"
        lines=$((lines + 1))
    fi

    # Summary line
    echo "-------------------------------------------------------------------------"
    lines=$((lines + 1))

    local running=$((LAUNCHED - COMPLETED - FAILED))
    local fail_str=""
    if [ $FAILED -gt 0 ]; then
        fail_str=" ($FAILED failed)"
    fi

    local now
    now=$(date +%s)
    local elapsed=$((now - START_TIME))
    local elapsed_min=$((elapsed / 60))
    local elapsed_sec=$((elapsed % 60))

    printf "  Done: %d/%d%s | Running: %d | Queued: %d | Time: %dm%02ds\n" \
        "$((COMPLETED + FAILED))" "$TOTAL" "$fail_str" "$running" "$queued" \
        "$elapsed_min" "$elapsed_sec"
    lines=$((lines + 1))

    DASHBOARD_LINES=$lines
}

# ─── Launch a job ───────────────────────────────────────────────────────────
launch_job() {
    local idx=$1
    local ckpt="${EVAL_CKPTS[$idx]}"
    local dirbase
    dirbase=$(basename "$(dirname "$ckpt")")
    local name
    name=$(basename "$ckpt" .pth)
    local key="${dirbase}/${name}"
    local logfile="$LOGDIR/${dirbase}_${name}_all.log"

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
    CUDA_VISIBLE_DEVICES=$best_gpu python run_lm_eval.py $COMMON_ARGS \
        --path "$ckpt" --tasks "$ALL_TASKS" \
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
                    FAILED=$((FAILED + 1))
                fi
            fi
        fi
    done
}

# ─── Fill GPU slots ────────────────────────────────────────────────────────
fill_slots() {
    while [ $NEXT_IDX -lt $TOTAL ]; do
        # Find a GPU with available slots
        local launched_any=0
        for ((g=0; g<NUM_GPUS; g++)); do
            if [ "${GPU_RUNNING[$g]}" -lt $JOBS_PER_GPU ] && [ $NEXT_IDX -lt $TOTAL ]; then
                launch_job $NEXT_IDX
                NEXT_IDX=$((NEXT_IDX + 1))
                launched_any=1
            fi
        done
        # If no slots available, break
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
    echo "Background processes killed. Logs so far saved to $LOGDIR/"
    exit 130
}
trap cleanup INT TERM

# ─── Main loop ──────────────────────────────────────────────────────────────
START_TIME=$(date +%s)

echo "Launching evaluations..."
echo ""

# Initial fill
fill_slots

while [ $((COMPLETED + FAILED)) -lt $TOTAL ]; do
    sleep 3
    check_finished
    fill_slots
    draw_dashboard
done

# Final dashboard
check_finished
draw_dashboard

# ─── Final summary ──────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo "========================================================================="
echo "                        EVALUATION COMPLETE                              "
echo "========================================================================="
echo ""
echo "  Total time:    ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "  Completed:     $COMPLETED / $TOTAL"
if [ $FAILED -gt 0 ]; then
    echo "  Failed:        $FAILED"
fi
echo "  Logs:          $LOGDIR/"
echo ""

# ─── Print results table ───────────────────────────────────────────────────
echo "========================================================================="
printf "  %-40s %s\n" "Model" "Results"
echo "-------------------------------------------------------------------------"

for ckpt in "${EVAL_CKPTS[@]}"; do
    dirbase=$(basename "$(dirname "$ckpt")")
    name=$(basename "$ckpt" .pth)
    logfile="$LOGDIR/${dirbase}_${name}_all.log"
    key="${dirbase}/${name}"

    echo ""
    echo "  -- $key --"

    if [ -f "$logfile" ]; then
        # Extract the final results dict (Python dict printed by run_lm_eval.py)
        # Try table format first
        grep -E '^\|.*\|.*\|.*acc' "$logfile" 2>/dev/null | while IFS= read -r line; do
            echo "    $line"
        done || true

        # If no table, try dict output
        if ! grep -qE '^\|.*\|.*\|.*acc' "$logfile" 2>/dev/null; then
            tail -5 "$logfile" 2>/dev/null | while IFS= read -r line; do
                echo "    $line"
            done || true
        fi
    else
        echo "    [no log file]"
    fi
done

echo ""
echo "========================================================================="

# Also run eval_manager summary if available
if [ -f "eval_manager.py" ]; then
    echo ""
    echo "Generating markdown summary..."
    python eval_manager.py summary 2>/dev/null || true
fi

echo ""
echo "Done! All logs saved to $LOGDIR/"
