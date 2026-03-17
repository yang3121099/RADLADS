#!/bin/bash
# Hybrid parallel eval: models distributed across GPUs + tasks parallelized per model
# 2 GPUs, each model gets 3 parallel task groups on its assigned GPU
# Models assigned round-robin to GPUs, all launched concurrently
#
# Usage:
#   bash run_eval_parallel.sh                    # eval ALL dirs under out/
#   bash run_eval_parallel.sh out/some_dir       # eval one dir only

set -e

NUM_GPUS=2
BSZ=${BSZ:-1}
COMMON="-c configs/qwen7b.yaml -c configs/qwerky7.yaml --model.attention_type rwkv7_fla_chunk --model.ctx_len 4096 --precision bf16 --bsz $BSZ"

# 3 task groups per model (balanced by workload)
TASKS_G1="hellaswag,lambada_openai"
TASKS_G2="arc_challenge,arc_easy,openbookqa"
TASKS_G3="boolq,piqa,winogrande"

# Collect all checkpoint dirs
if [ -n "$1" ]; then
    CKPT_DIRS=("$1")
else
    CKPT_DIRS=(out/*)
fi

# Collect all checkpoints
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
echo "Strategy: models round-robin on $NUM_GPUS GPUs, 3 task groups per model"
echo "  G1: $TASKS_G1"
echo "  G2: $TASKS_G2"
echo "  G3: $TASKS_G3"
echo "Batch size: $BSZ"
echo "=========================================="

# Launch ALL models concurrently, round-robin GPU assignment
ALL_PIDS=()
ALL_LABELS=()
ALL_LOGDIRS=()
ALL_NAMES=()

gpu=0
for ckpt in "${ALL_CKPTS[@]}"; do
    dirbase=$(basename "$(dirname "$ckpt")")
    name=$(basename "$ckpt" .pth)
    LOGDIR="eval_logs/$dirbase"
    mkdir -p "$LOGDIR"

    echo ""
    echo "=== [GPU $gpu] $dirbase / $name ==="

    # Launch 3 task groups on this GPU
    CUDA_VISIBLE_DEVICES=$gpu python run_lm_eval.py $COMMON --path "$ckpt" \
        --tasks "$TASKS_G1" \
        > "$LOGDIR/${name}_g1.log" 2>&1 &
    ALL_PIDS+=($!)

    CUDA_VISIBLE_DEVICES=$gpu python run_lm_eval.py $COMMON --path "$ckpt" \
        --tasks "$TASKS_G2" \
        > "$LOGDIR/${name}_g2.log" 2>&1 &
    ALL_PIDS+=($!)

    CUDA_VISIBLE_DEVICES=$gpu python run_lm_eval.py $COMMON --path "$ckpt" \
        --tasks "$TASKS_G3" \
        > "$LOGDIR/${name}_g3.log" 2>&1 &
    ALL_PIDS+=($!)

    ALL_LABELS+=("$dirbase/$name" "$dirbase/$name" "$dirbase/$name")
    ALL_LOGDIRS+=("$LOGDIR" "$LOGDIR" "$LOGDIR")
    ALL_NAMES+=("$name" "$name" "$name")

    echo "  G1: $TASKS_G1"
    echo "  G2: $TASKS_G2"
    echo "  G3: $TASKS_G3"

    # Round-robin GPU
    gpu=$(( (gpu + 1) % NUM_GPUS ))
done

total_procs=${#ALL_PIDS[@]}
total_models=${#ALL_CKPTS[@]}
echo ""
echo "=========================================="
echo "All $total_models models launched ($total_procs processes total)"
echo "Waiting for completion..."

# Wait for all processes
FAIL=0
for j in "${!ALL_PIDS[@]}"; do
    if ! wait "${ALL_PIDS[$j]}"; then
        echo "  [FAIL] ${ALL_LABELS[$j]} (group $((j % 3 + 1)))"
        FAIL=$((FAIL+1))
    fi
done

# Print results per model
echo ""
echo "=========================================="
echo "Results:"
SEEN=()
for ckpt in "${ALL_CKPTS[@]}"; do
    dirbase=$(basename "$(dirname "$ckpt")")
    name=$(basename "$ckpt" .pth)
    LOGDIR="eval_logs/$dirbase"
    echo ""
    echo "--- $dirbase / $name ---"
    for log in "$LOGDIR/${name}"_g*.log; do
        grep -E "^\|.*\|.*\|.*acc" "$log" 2>/dev/null || true
    done
    echo "--------------------------------"
done

if [ $FAIL -gt 0 ]; then
    echo ""
    echo "[WARN] $FAIL process(es) failed, check logs"
fi

echo ""
echo "All $total_models models evaluated! Logs: eval_logs/"
