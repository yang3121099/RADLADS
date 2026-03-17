#!/bin/bash
# ============================================================================
# run_eval_tqa_gpqa.sh - 测评 truthfulqa_mc2 和 gpqa_diamond_zeroshot
#
# 对 out/ 下所有模型测评，**排除** 6个 chatbot SFT 模型。
# 2 GPUs × JOBS_PER_GPU 并行，自动跳过已完成的模型。
# 完成后生成 markdown 汇总表格。
#
# Usage:
#   bash run_eval_tqa_gpqa.sh                   # 默认 JOBS_PER_GPU=1, BSZ=16
#   JOBS_PER_GPU=2 BSZ=8 bash run_eval_tqa_gpqa.sh
#   bash run_eval_tqa_gpqa.sh --force           # 强制重测
# ============================================================================

set -uo pipefail

# ─── Configuration ──────────────────────────────────────────────────────────
NUM_GPUS=${NUM_GPUS:-2}
JOBS_PER_GPU=${JOBS_PER_GPU:-1}
BSZ=${BSZ:-16}
FORCE=0
LOGDIR="eval_logs/tqa_gpqa"
TASKS="truthfulqa_mc2,gpqa_diamond_zeroshot"
RESULTS_JSON="eval_tqa_gpqa_results.json"

for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        *) echo "[ERROR] Unknown argument: $arg (use: --force)"; exit 1 ;;
    esac
done

# ─── 6 models to EXCLUDE (chatbot SFT models) ──────────────────────────────
EXCLUDE_MODELS=(
    "out/L28-D3584-qwerky7_qwen2-6_chatbot_slimorca/rwkv-step1500-197M.pth"
    "out/L28-D3584-qwerky7_qwen2-6_chatbot_slimorca/rwkv-step150-20M.pth"
    "out/L28-D3584-qwerky7_qwen2-6_chatbot_openhermes/rwkv-step900-118M.pth"
    "out/L28-D3584-qwerky7_qwen2-6_chatbot_ultrachat/rwkv-step1500-197M.pth"
    "out/L28-D3584-qwerky7_qwen2-6_chatbot_ultrachat/rwkv-step150-20M.pth"
    "out/L28-D3584-qwerky7_qwen2-4_BOA/rwkv-1.pth"
)

# Build exclusion set (absolute paths)
declare -A EXCLUDE_SET
for m in "${EXCLUDE_MODELS[@]}"; do
    abs=$(realpath "$m" 2>/dev/null || echo "$m")
    EXCLUDE_SET["$abs"]=1
done

# ─── Discover all models ────────────────────────────────────────────────────
mapfile -t ALL_MODELS < <(find out/ -name "rwkv-*.pth" 2>/dev/null | sort)

MODELS=()
SKIPPED_EXCLUDE=0
for m in "${ALL_MODELS[@]}"; do
    abs=$(realpath "$m" 2>/dev/null || echo "$m")
    if [[ -v "EXCLUDE_SET[$abs]" ]]; then
        SKIPPED_EXCLUDE=$((SKIPPED_EXCLUDE + 1))
        continue
    fi
    MODELS+=("$m")
done

TOTAL=${#MODELS[@]}

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║     TruthfulQA_MC2 + GPQA_Diamond Batch Evaluation                ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Tasks:         $TASKS"
echo "  GPUs:          $NUM_GPUS"
echo "  Jobs per GPU:  $JOBS_PER_GPU"
echo "  Max parallel:  $((NUM_GPUS * JOBS_PER_GPU))"
echo "  Batch size:    $BSZ"
echo "  Force re-eval: $( [ $FORCE -eq 1 ] && echo 'YES' || echo 'no' )"
echo "  Total found:   ${#ALL_MODELS[@]}"
echo "  Excluded:      $SKIPPED_EXCLUDE (chatbot SFT models)"
echo "  To evaluate:   $TOTAL"
echo "  Results log:   $RESULTS_JSON"
echo ""

if [ $TOTAL -eq 0 ]; then
    echo "[ERROR] No models found to evaluate."
    exit 1
fi

for m in "${MODELS[@]}"; do
    echo "  [OK] $m"
done
echo ""

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
    key="${dirbase}/${name}"
    logfile="$LOGDIR/${dirbase}_${name}.log"

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
        --group chatbot \
        --tasks "$TASKS" \
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

# ─── Collect results and generate markdown summary ─────────────────────────
echo "Generating markdown summary..."
python3 - "$RESULTS_JSON" <<'PYTHON_SCRIPT'
import json, os, sys, re, glob

RESULTS_JSON = sys.argv[1] if len(sys.argv) > 1 else "eval_tqa_gpqa_results.json"
SUMMARY_MD = RESULTS_JSON.replace(".json", "_summary.md")

# ─── Load eval_chatbot_results.json (primary source) ──────────────────────
chatbot_log = {}
if os.path.exists("eval_chatbot_results.json"):
    with open("eval_chatbot_results.json") as f:
        chatbot_log = json.load(f)

# ─── Also load eval_results.json for models tracked by eval_manager ───────
manager_log = {}
if os.path.exists("eval_results.json"):
    with open("eval_results.json") as f:
        manager_log = json.load(f)

# ─── 6 models to exclude ──────────────────────────────────────────────────
EXCLUDE = {
    os.path.abspath(p) for p in [
        "out/L28-D3584-qwerky7_qwen2-6_chatbot_slimorca/rwkv-step1500-197M.pth",
        "out/L28-D3584-qwerky7_qwen2-6_chatbot_slimorca/rwkv-step150-20M.pth",
        "out/L28-D3584-qwerky7_qwen2-6_chatbot_openhermes/rwkv-step900-118M.pth",
        "out/L28-D3584-qwerky7_qwen2-6_chatbot_ultrachat/rwkv-step1500-197M.pth",
        "out/L28-D3584-qwerky7_qwen2-6_chatbot_ultrachat/rwkv-step150-20M.pth",
        "out/L28-D3584-qwerky7_qwen2-4_BOA/rwkv-1.pth",
    ] if os.path.exists(p)
}

# ─── Merge results ────────────────────────────────────────────────────────
# Collect all model entries with truthfulqa_mc2 or gpqa_diamond_zeroshot results
entries = []

for key, entry in {**manager_log, **chatbot_log}.items():
    # Skip excluded models
    if key in EXCLUDE or entry.get("path", "") in EXCLUDE:
        abs_path = os.path.abspath(entry.get("path", ""))
        if abs_path in EXCLUDE:
            continue
    if key.startswith("baseline:"):
        continue

    # Get results from either format
    results = {}
    if "results" in entry and isinstance(entry["results"], dict):
        results = entry["results"]
    if "standard_results" in entry and isinstance(entry["standard_results"], dict):
        results.update(entry["standard_results"])

    tqa = results.get("truthfulqa_mc2")
    gpqa = results.get("gpqa_diamond_zeroshot")

    if tqa is None and gpqa is None:
        continue

    # Extract display info
    basename = entry.get("basename", os.path.basename(entry.get("path", "?")))
    dirname = entry.get("dir", os.path.basename(os.path.dirname(entry.get("path", ""))))
    step = entry.get("step", 0)
    label = entry.get("label", "?")

    entries.append({
        "dirname": dirname,
        "basename": basename,
        "step": step,
        "label": label,
        "truthfulqa_mc2": tqa,
        "gpqa_diamond_zeroshot": gpqa,
    })

# Sort by dirname then step
entries.sort(key=lambda x: (x["dirname"], x["step"]))

# ─── Generate markdown table ──────────────────────────────────────────────
lines = []
lines.append("# TruthfulQA_MC2 + GPQA Diamond Evaluation Results\n")
lines.append(f"Total models: {len(entries)} (excluding 6 chatbot SFT models)\n")

lines.append("| Model | Tokens | truthfulqa_mc2 | gpqa_diamond | avg |")
lines.append("|---|---|---|---|---|")

for e in entries:
    short_dir = e["dirname"]
    # Shorten common prefixes
    for prefix in ["L28-D3584-qwerky7_qwen2-", "L28-D3584-"]:
        if short_dir.startswith(prefix):
            short_dir = short_dir[len(prefix):]
            break
    display = f"{short_dir}/{e['basename']}" if short_dir else e["basename"]

    tqa = e["truthfulqa_mc2"]
    gpqa = e["gpqa_diamond_zeroshot"]

    tqa_str = f"{tqa:.1f}" if tqa is not None else "-"
    gpqa_str = f"{gpqa:.1f}" if gpqa is not None else "-"

    vals = [v for v in [tqa, gpqa] if v is not None]
    avg = sum(vals) / len(vals) if vals else 0
    avg_str = f"**{avg:.1f}**" if vals else "-"

    lines.append(f"| {display} | {e['label']} | {tqa_str} | {gpqa_str} | {avg_str} |")

md = "\n".join(lines) + "\n"

# Write summary
with open(SUMMARY_MD, "w") as f:
    f.write(md)

# Also print to stdout
print()
print(md)
print(f"Summary saved to {SUMMARY_MD}")

# Save structured results
with open(RESULTS_JSON, "w") as f:
    json.dump(entries, f, indent=2, ensure_ascii=False)
print(f"Results saved to {RESULTS_JSON}")
PYTHON_SCRIPT

echo ""
echo "Done! See eval_tqa_gpqa_results_summary.md for the full table."
