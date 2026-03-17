#!/bin/bash
# Model-parallel eval: 2 GPUs × 3 models each = up to 6 models at once
# Each model runs ALL tasks in one process sequentially.
# Tasks: arc_c, arc_e, boolq, hellaswag, lambada, obqa, piqa, winogrande
#
# Usage:
#   bash run_eval_parallel.sh                    # eval ALL dirs under out/
#   bash run_eval_parallel.sh out/some_dir       # eval one dir only

set -e

NUM_GPUS=2
PROCS_PER_GPU=3
SLOTS=$((NUM_GPUS * PROCS_PER_GPU))  # 6

BSZ=${BSZ:-1}
COMMON="-c configs/qwen7b.yaml -c configs/qwerky7.yaml --model.attention_type rwkv7_fla_chunk --model.ctx_len 4096 --precision bf16 --bsz $BSZ"
ALL_TASKS="hellaswag,lambada_openai,boolq,arc_challenge,arc_easy,openbookqa,piqa,winogrande"

# Collect all checkpoint dirs
if [ -n "$1" ]; then
    CKPT_DIRS=("$1")
else
    CKPT_DIRS=(out/*)
fi

# Collect all checkpoints across all dirs
ALL_CKPTS=()
for dir in "${CKPT_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        for ckpt in "$dir"/rwkv-*.pth; do
            [ -f "$ckpt" ] && ALL_CKPTS+=("$ckpt")
        done
    fi
done

if [ ${#ALL_CKPTS[@]} -eq 0 ]; then
    echo "[ERROR] No checkpoints found"
    exit 1
fi

echo "Found ${#ALL_CKPTS[@]} checkpoint(s) across ${#CKPT_DIRS[@]} dir(s):"
for dir in "${CKPT_DIRS[@]}"; do
    count=$(ls "$dir"/rwkv-*.pth 2>/dev/null | wc -l)
    echo "  $dir: $count checkpoint(s)"
done
echo "Strategy: $NUM_GPUS GPUs × $PROCS_PER_GPU models/GPU = $SLOTS concurrent models"
echo "Each model runs all tasks: $ALL_TASKS"
echo "Batch size: $BSZ"
echo "=========================================="

# --- Launch models in batches of $SLOTS ---
total=${#ALL_CKPTS[@]}
idx=0

while [ $idx -lt $total ]; do
    PIDS=()
    LABELS=()
    batch_end=$((idx + SLOTS))
    [ $batch_end -gt $total ] && batch_end=$total

    echo ""
    echo ">>> Batch: checkpoints $((idx+1))-${batch_end} of $total"

    slot=0
    for (( i=idx; i<batch_end; i++ )); do
        ckpt="${ALL_CKPTS[$i]}"
        dirbase=$(basename "$(dirname "$ckpt")")
        name=$(basename "$ckpt" .pth)
        gpu=$((slot / PROCS_PER_GPU))
        LOGDIR="eval_logs/$dirbase"
        mkdir -p "$LOGDIR"
        logfile="$LOGDIR/${name}_all.log"

        echo "  [GPU $gpu] $dirbase / $name -> $logfile"

        CUDA_VISIBLE_DEVICES=$gpu python run_lm_eval.py $COMMON --path "$ckpt" \
            --tasks "$ALL_TASKS" \
            > "$logfile" 2>&1 &
        PIDS+=($!)
        LABELS+=("$dirbase/$name")
        slot=$((slot + 1))
    done

    echo "  Waiting for ${#PIDS[@]} model(s)..."

    FAIL=0
    for j in "${!PIDS[@]}"; do
        if ! wait "${PIDS[$j]}"; then
            echo "  [FAIL] ${LABELS[$j]}"
            FAIL=$((FAIL+1))
        else
            echo "  [OK]   ${LABELS[$j]}"
        fi
    done

    # Print results for this batch
    for (( i=idx; i<batch_end; i++ )); do
        ckpt="${ALL_CKPTS[$i]}"
        dirbase=$(basename "$(dirname "$ckpt")")
        name=$(basename "$ckpt" .pth)
        logfile="eval_logs/$dirbase/${name}_all.log"
        echo ""
        echo "--- Results: $dirbase / $name ---"
        grep -E "^\|.*\|.*\|.*acc" "$logfile" 2>/dev/null || echo "  (no results, check log)"
        echo "--------------------------------"
    done

    idx=$batch_end
done

echo ""
echo "=========================================="
echo "All $total checkpoints evaluated!"
echo "Logs saved to: eval_logs/"
