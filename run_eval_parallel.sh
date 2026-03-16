#!/bin/bash
# Parallel eval: 2 GPUs × 3 processes each = 6 processes
# Tasks: arc_c, arc_e, boolq, hella, lambada, obqa, piqa, wino
#
# Usage:
#   bash run_eval_parallel.sh out/L28-D3584-qwerky7_qwen2-6_chatbot_openhermes

set -e

CKPT_DIR="${1:?Usage: bash run_eval_parallel.sh <checkpoint_dir>}"
BSZ=${2:-1}

COMMON="-c configs/qwen7b.yaml -c configs/qwerky7.yaml --model.attention_type rwkv7_fla_chunk --model.ctx_len 4096 --precision bf16 --bsz $BSZ"

CKPTS=($(ls "$CKPT_DIR"/rwkv-*.pth 2>/dev/null | sort))
if [ ${#CKPTS[@]} -eq 0 ]; then
    echo "[ERROR] No checkpoints found in $CKPT_DIR"
    exit 1
fi

echo "Found ${#CKPTS[@]} checkpoint(s) in $CKPT_DIR"
echo "Batch size: $BSZ"
echo "=========================================="

LOGDIR="eval_logs/$(basename $CKPT_DIR)"
mkdir -p "$LOGDIR"

for ckpt in "${CKPTS[@]}"; do
    name=$(basename "$ckpt" .pth)
    echo ""
    echo "=== Evaluating: $name ==="
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
echo "All checkpoints evaluated!"
echo "Logs saved to: $LOGDIR/"
