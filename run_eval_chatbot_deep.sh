#!/bin/bash
# ============================================================================
# run_eval_chatbot_deep.sh - Chatbot 深度测评 (社交/安全/知识/推理)
#
# 先跑 6 个 chatbot SFT 模型，支持后续全量模型。
# 2 GPUs × JOBS_PER_GPU 并行，自动跳过已完成。
#
# 任务分组:
#   chatbot_core  : social_iqa, ethics_utilitarianism, ethics_justice,
#                   toxigen, crows_pairs_english
#   chatbot_extra : commonsense_qa, sciq, logiqa, anli_r3
#   chatbot_deep  : chatbot_core + chatbot_extra (全部)
#
# Usage:
#   bash run_eval_chatbot_deep.sh                        # 默认: 6模型, chatbot_deep
#   bash run_eval_chatbot_deep.sh chatbot_core           # 仅社交/安全
#   bash run_eval_chatbot_deep.sh chatbot_extra          # 仅知识/推理
#   bash run_eval_chatbot_deep.sh --all-models           # 全量模型 (排除6个chatbot)
#   bash run_eval_chatbot_deep.sh --force                # 强制重测
#   JOBS_PER_GPU=2 BSZ=8 bash run_eval_chatbot_deep.sh  # 自定义并行度
# ============================================================================

set -uo pipefail

# ─── Configuration ──────────────────────────────────────────────────────────
NUM_GPUS=${NUM_GPUS:-2}
JOBS_PER_GPU=${JOBS_PER_GPU:-1}
BSZ=${BSZ:-16}
FORCE=0
LOGDIR="eval_logs/chatbot_deep"
EVAL_GROUP="chatbot_deep"
ALL_MODELS_MODE=0

for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        --all-models) ALL_MODELS_MODE=1 ;;
        chatbot_core|chatbot_extra|chatbot_deep) EVAL_GROUP="$arg" ;;
        *) echo "[ERROR] Unknown argument: $arg"; echo "Usage: bash run_eval_chatbot_deep.sh [chatbot_core|chatbot_extra|chatbot_deep] [--all-models] [--force]"; exit 1 ;;
    esac
done

# ─── 6 chatbot SFT models ──────────────────────────────────────────────────
SFT_MODELS=(
    "out/L28-D3584-qwerky7_qwen2-6_chatbot_slimorca/rwkv-step1500-197M.pth"
    "out/L28-D3584-qwerky7_qwen2-6_chatbot_slimorca/rwkv-step150-20M.pth"
    "out/L28-D3584-qwerky7_qwen2-6_chatbot_openhermes/rwkv-step900-118M.pth"
    "out/L28-D3584-qwerky7_qwen2-6_chatbot_ultrachat/rwkv-step1500-197M.pth"
    "out/L28-D3584-qwerky7_qwen2-6_chatbot_ultrachat/rwkv-step150-20M.pth"
    "out/L28-D3584-qwerky7_qwen2-4_BOA/rwkv-1.pth"
)

# ─── Model selection ────────────────────────────────────────────────────────
if [ $ALL_MODELS_MODE -eq 1 ]; then
    # All models in out/, EXCLUDING the 6 SFT models
    declare -A EXCLUDE_SET
    for m in "${SFT_MODELS[@]}"; do
        abs=$(realpath "$m" 2>/dev/null || echo "$m")
        EXCLUDE_SET["$abs"]=1
    done
    mapfile -t ALL_FOUND < <(find out/ -name "rwkv-*.pth" 2>/dev/null | sort)
    MODELS=()
    for m in "${ALL_FOUND[@]}"; do
        abs=$(realpath "$m" 2>/dev/null || echo "$m")
        [[ -v "EXCLUDE_SET[$abs]" ]] && continue
        MODELS+=("$m")
    done
    MODE_LABEL="all models (excluding 6 SFT)"
else
    MODELS=("${SFT_MODELS[@]}")
    MODE_LABEL="6 chatbot SFT models"
fi

TOTAL=${#MODELS[@]}

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║        Chatbot Deep Evaluation — 社交/安全/知识/推理              ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Group:         $EVAL_GROUP"
echo "  Mode:          $MODE_LABEL"
echo "  GPUs:          $NUM_GPUS"
echo "  Jobs per GPU:  $JOBS_PER_GPU"
echo "  Max parallel:  $((NUM_GPUS * JOBS_PER_GPU))"
echo "  Batch size:    $BSZ"
echo "  Force re-eval: $( [ $FORCE -eq 1 ] && echo 'YES' || echo 'no' )"
echo "  Models:        $TOTAL"
echo ""

if [ $TOTAL -eq 0 ]; then
    echo "[ERROR] No models found to evaluate."
    exit 1
fi

# Verify models exist
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

# ─── Job tracking ──────────────────────────────────────────────────────────
declare -A JOB_PIDS
declare -A JOB_GPUS
declare -A JOB_LOGS
declare -A JOB_LABELS
declare -A JOB_STATUS
declare -A GPU_RUNNING

for ((g=0; g<NUM_GPUS; g++)); do
    GPU_RUNNING[$g]=0
done

LAUNCHED=0
COMPLETED=0
FAILED_COUNT=0
NEXT_IDX=0

# ─── Progress display ──────────────────────────────────────────────────────
get_progress() {
    local logfile="$1"
    [ ! -f "$logfile" ] && echo "-1" && return 0
    local tail_text
    tail_text=$(tail -c 4096 "$logfile" 2>/dev/null || true)
    local pct
    pct=$(echo "$tail_text" | grep -oE '[0-9]+%' | tail -1 | grep -oE '[0-9]+' || true)
    [ -n "$pct" ] && echo "$pct" && return 0
    echo "$tail_text" | grep -qE 'Loading model|RWKV_MODEL_TYPE' 2>/dev/null && echo "0" && return 0
    echo "$tail_text" | grep -qE '\[OK\]|\[SKIP\]' 2>/dev/null && echo "50" && return 0
    echo "-1"
    return 0
}

make_bar() {
    local pct=$1 width=25
    local filled=$((width * pct / 100))
    local empty=$((width - filled))
    local bar="" i
    for ((i=0; i<filled; i++)); do bar+="="; done
    for ((i=0; i<empty; i++)); do bar+="-"; done
    echo "$bar"
}

DASHBOARD_LINES=0
draw_dashboard() {
    if [ $DASHBOARD_LINES -gt 0 ]; then
        printf '\033[%dA\033[J' "$DASHBOARD_LINES"
    fi
    local lines=1
    echo "-------------------------------------------------------------------------"
    lines=$((lines + 1))
    printf "  %-40s %-6s %s\n" "Model" "GPU" "Progress"
    lines=$((lines + 1))
    echo "-------------------------------------------------------------------------"
    lines=$((lines + 1))

    local sorted_keys
    sorted_keys=$(echo "${!JOB_STATUS[@]}" | tr ' ' '\n' | sort)
    for key in $sorted_keys; do
        local status="${JOB_STATUS[$key]}"
        local gpu="${JOB_GPUS[$key]}"
        local label="${JOB_LABELS[$key]}"
        local logfile="${JOB_LOGS[$key]}"
        if [ ${#label} -gt 40 ]; then label="${label:0:37}..."; fi
        if [ "$status" = "running" ]; then
            local pct bar
            pct=$(get_progress "$logfile")
            if [ -z "$pct" ] || [ "$pct" = "-1" ]; then
                printf "  %-40s GPU%-3s > [...........................] ...\n" "$label" "$gpu"
            else
                bar=$(make_bar "$pct")
                printf "  %-40s GPU%-3s > [%s] %3d%%\n" "$label" "$gpu" "$bar" "$pct"
            fi
        elif [ "$status" = "done" ]; then
            printf "  %-40s GPU%-3s   [=========================] done\n" "$label" "$gpu"
        elif [ "$status" = "fail" ]; then
            printf "  %-40s GPU%-3s   FAILED\n" "$label" "$gpu"
        fi
        lines=$((lines + 1))
    done

    echo "-------------------------------------------------------------------------"
    lines=$((lines + 1))
    local running=$((LAUNCHED - COMPLETED - FAILED_COUNT))
    local fail_str=""
    [ $FAILED_COUNT -gt 0 ] && fail_str=" ($FAILED_COUNT failed)"
    local now elapsed elapsed_min elapsed_sec
    now=$(date +%s)
    elapsed=$((now - START_TIME))
    elapsed_min=$((elapsed / 60))
    elapsed_sec=$((elapsed % 60))
    printf "  Done: %d/%d%s | Running: %d | Time: %dm%02ds\n" \
        "$((COMPLETED + FAILED_COUNT))" "$TOTAL" "$fail_str" "$running" "$elapsed_min" "$elapsed_sec"
    lines=$((lines + 1))
    DASHBOARD_LINES=$lines
}

# ─── Launch a job ──────────────────────────────────────────────────────────
launch_job() {
    local idx=$1
    local model="${MODELS[$idx]}"
    local dirbase name key logfile
    dirbase=$(basename "$(dirname "$model")")
    name=$(basename "$model" .pth)
    key="${dirbase##*_}/${name}"
    logfile="$LOGDIR/${dirbase}_${name}_${EVAL_GROUP}.log"

    # Pick GPU with fewest running jobs
    local best_gpu=0 min_jobs=${GPU_RUNNING[0]}
    for ((g=1; g<NUM_GPUS; g++)); do
        if [ "${GPU_RUNNING[$g]}" -lt "$min_jobs" ]; then
            min_jobs=${GPU_RUNNING[$g]}
            best_gpu=$g
        fi
    done

    local force_flag=""
    [ $FORCE -eq 1 ] && force_flag="--force"

    CUDA_VISIBLE_DEVICES=$best_gpu python eval_chatbot.py eval \
        --path "$model" \
        --bsz $BSZ \
        --group "$EVAL_GROUP" \
        $force_flag \
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

# ─── Check finished jobs ──────────────────────────────────────────────────
check_finished() {
    for key in "${!JOB_PIDS[@]}"; do
        if [ "${JOB_STATUS[$key]}" = "running" ]; then
            local pid=${JOB_PIDS[$key]}
            if ! kill -0 "$pid" 2>/dev/null; then
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

# ─── Fill GPU slots ───────────────────────────────────────────────────────
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
        [ $launched_any -eq 0 ] && break
    done
}

# ─── Cleanup ──────────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[INTERRUPTED] Killing running eval processes..."
    for key in "${!JOB_PIDS[@]}"; do
        [ "${JOB_STATUS[$key]}" = "running" ] && kill "${JOB_PIDS[$key]}" 2>/dev/null || true
    done
    echo "Logs in $LOGDIR/"
    exit 130
}
trap cleanup INT TERM

# ─── Main loop ─────────────────────────────────────────────────────────────
START_TIME=$(date +%s)
echo "Launching evaluations..."
echo ""

fill_slots

while [ $((COMPLETED + FAILED_COUNT)) -lt $TOTAL ]; do
    sleep 3
    check_finished
    fill_slots
    draw_dashboard
done
draw_dashboard

# ─── Final summary ─────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo "========================================================================="
echo "                        EVALUATION COMPLETE"
echo "========================================================================="
echo "  Group:       $EVAL_GROUP"
echo "  Total time:  ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "  Completed:   $((TOTAL - FAILED_COUNT)) / $TOTAL"
if [ $FAILED_COUNT -gt 0 ]; then
    echo "  Failed:      $FAILED_COUNT"
    echo ""
    echo "  Failed model logs:"
    for key in "${!JOB_STATUS[@]}"; do
        [ "${JOB_STATUS[$key]}" = "fail" ] && echo "    - ${JOB_LOGS[$key]}"
    done
fi
echo ""

# Generate summary
echo "Generating summary..."
python eval_chatbot.py summary
echo ""
echo "Done! Results in eval_chatbot_results.json / eval_chatbot_summary.md"
