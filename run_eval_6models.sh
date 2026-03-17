#!/bin/bash
# ============================================================================
# run_eval_6models.sh - 6个 chatbot SFT 模型全量测评
#
# 分两阶段:
#   Phase 1 (fast): loglikelihood 指标 — lambada, hellaswag, winogrande, piqa,
#                   truthfulqa_mc2, arc_challenge, mmlu, boolq, mmlu_pro, gpqa
#   Phase 2 (slow): generate_until 指标 — gsm8k, ifeval, bbh_zeroshot
#
# Usage:
#   bash run_eval_6models.sh               # 全量 (fast + slow)
#   bash run_eval_6models.sh fast           # 仅跑 loglikelihood 指标
#   bash run_eval_6models.sh generative     # 仅跑 generate_until 指标
#   GPU=0 bash run_eval_6models.sh          # 指定 GPU
# ============================================================================
set -uo pipefail

MODE=${1:-all}      # all, fast, generative
GPU=${GPU:-0}
BSZ=8

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
echo "║              RADLADS 6-Model Full Benchmark Evaluation             ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Mode:       $MODE"
echo "  GPU:        $GPU"
echo "  Batch size: $BSZ"
echo "  Models:     ${#MODELS[@]}"
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

# Determine groups to run
if [ "$MODE" = "all" ]; then
    GROUPS="all"
elif [ "$MODE" = "fast" ]; then
    GROUPS="fast"
elif [ "$MODE" = "generative" ]; then
    GROUPS="generative"
else
    echo "[ERROR] Unknown mode: $MODE (use: all, fast, generative)"
    exit 1
fi

START_TIME=$(date +%s)
TOTAL=${#MODELS[@]}
DONE=0
FAILED=0

for model in "${MODELS[@]}"; do
    DONE=$((DONE + 1))
    dirbase=$(basename "$(dirname "$model")")
    name=$(basename "$model" .pth)
    short="${dirbase##*_}/$name"

    echo ""
    echo "========================================================================="
    echo "  [$DONE/$TOTAL] $short"
    echo "  Path: $model"
    echo "  Group: $GROUPS"
    echo "========================================================================="
    echo ""

    CUDA_VISIBLE_DEVICES=$GPU python eval_chatbot.py eval \
        --path "$model" \
        --bsz $BSZ \
        --gpu $GPU \
        --group "$GROUPS" \
    || {
        echo "[FAILED] $short"
        FAILED=$((FAILED + 1))
    }
done

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
echo "  Completed:   $((TOTAL - FAILED)) / $TOTAL"
if [ $FAILED -gt 0 ]; then
    echo "  Failed:      $FAILED"
fi
echo ""

# Generate final summary
echo "Generating summary..."
python eval_chatbot.py summary
echo ""
echo "Done! Results in eval_chatbot_results.json / eval_chatbot_summary.md"
