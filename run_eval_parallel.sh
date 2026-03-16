#!/bin/bash
# Parallel eval: 2 GPUs × 3 processes each = 6 processes
# Tasks: arc_c, arc_e, boolq, hella, lambada, obqa, piqa, wino
#
# Usage:
#   bash run_eval_parallel.sh                    # eval ALL dirs under out/
#   bash run_eval_parallel.sh out/some_dir       # eval one dir only

set -e

BSZ=${BSZ:-1}
COMMON="-c configs/qwen7b.yaml -c configs/qwerky7.yaml --model.attention_type rwkv7_fla_chunk --model.ctx_len 4096 --precision bf16 --bsz $BSZ"

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
echo "Batch size: $BSZ"
echo "=========================================="

for ckpt in "${ALL_CKPTS[@]}"; do
    dirname=$(basename "$(dirname "$ckpt")")
    name=$(basename "$ckpt" .pth)
    LOGDIR="eval_logs/$dirname"
    mkdir -p "$LOGDIR"

    echo ""
    echo "=== [$dirname] Evaluating: $name ==="
    echo ""

    # GPU 0 - 3 processes (heavier tasks get their own process)
    CUDA_VISIBLE_DEVICES=0 python run_lm_eval.py $COMMON --path "$ckpt" \
        --tasks hellaswag \
        > "$LOGDIR/${name}_gpu0_p1.log" 2>&1 &
    p1=$!

    CUDA_VISIBLE_DEVICES=0 python run_lm_eval.py $COMMON --path "$ckpt" \
        --tasks lambada_openai,boolq \
        > "$LOGDIR/${name}_gpu0_p2.log" 2>&1 &
    p2=$!

    CUDA_VISIBLE_DEVICES=0 python run_lm_eval.py $COMMON --path "$ckpt" \
        --tasks arc_challenge,openbookqa \
        > "$LOGDIR/${name}_gpu0_p3.log" 2>&1 &
    p3=$!

    # GPU 1 - 3 processes
    CUDA_VISIBLE_DEVICES=1 python run_lm_eval.py $COMMON --path "$ckpt" \
        --tasks arc_easy,piqa \
        > "$LOGDIR/${name}_gpu1_p1.log" 2>&1 &
    p4=$!

    CUDA_VISIBLE_DEVICES=1 python run_lm_eval.py $COMMON --path "$ckpt" \
        --tasks winogrande \
        > "$LOGDIR/${name}_gpu1_p2.log" 2>&1 &
    p5=$!

    # mmlu is not in the task list but GPU 1 proc 3 is free
    # if you want to add more tasks, uncomment:
    # CUDA_VISIBLE_DEVICES=1 python run_lm_eval.py $COMMON --path "$ckpt" \
    #     --tasks mmlu \
    #     > "$LOGDIR/${name}_gpu1_p3.log" 2>&1 &
    # p6=$!

    echo "  GPU0: hellaswag | lambada+boolq | arc_c+obqa"
    echo "  GPU1: arc_e+piqa | winogrande | (free)"
    echo "  Logs: $LOGDIR/${name}_*.log"
    echo "  Waiting for all 5 processes..."

    FAIL=0
    for pid in $p1 $p2 $p3 $p4 $p5; do
        wait $pid || FAIL=$((FAIL+1))
    done

    if [ $FAIL -gt 0 ]; then
        echo "  [WARN] $FAIL process(es) failed for $name, check logs"
    else
        echo "  [OK] All tasks done for $name"
    fi

    # Print summary from logs
    echo ""
    echo "--- Results for $name ---"
    for log in "$LOGDIR/${name}"_*.log; do
        grep -E "^\|.*\|.*\|.*acc" "$log" 2>/dev/null || true
    done
    echo "-------------------------"
done

echo ""
echo "=========================================="
echo "All ${#ALL_CKPTS[@]} checkpoints evaluated!"
echo "Logs saved to: eval_logs/"
