#!/bin/bash
# ============================================================================
# Qwen2.5-7B-Instruct 基线测评脚本
#
# 用法:
#   bash eval_qwen_baseline.sh                          # 默认 Qwen2.5-7B-Instruct
#   bash eval_qwen_baseline.sh Qwen/Qwen2-7B           # 指定其他模型
#   bash eval_qwen_baseline.sh Qwen/Qwen2.5-7B         # 非 Instruct 版本
#
# 说明:
#   使用 lm_eval 内置的 HuggingFace 后端直接加载原版 Qwen 模型，
#   测试与 RADLADS 相同的 base benchmark 任务，用于对比。
# ============================================================================
set -e

# === 配置 ===
MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_SHORT=$(echo "$MODEL" | sed 's|/|-|g')
BSZ="${BSZ:-auto}"
RESULTS_DIR="eval_results"
GPU="${GPU:-0}"

# === 任务 (与 eval_reasoning.sh 的 TASKS_BASE 保持一致) ===
TASKS="lambada_openai,arc_easy,arc_challenge,hellaswag,winogrande,piqa,openbookqa,boolq"

mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo " Qwen Baseline Evaluation"
echo " Model: $MODEL"
echo " Tasks: $TASKS"
echo " Batch: $BSZ"
echo " GPU:   $GPU"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL,dtype=float16,trust_remote_code=True" \
    --tasks $TASKS \
    --batch_size $BSZ \
    --device cuda \
    --output_path "${RESULTS_DIR}/${MODEL_SHORT}" \
    2>&1 | tee "${RESULTS_DIR}/${MODEL_SHORT}.log"

echo ""
echo "=========================================="
echo " 测评完成!"
echo " 结果: ${RESULTS_DIR}/${MODEL_SHORT}.log"
echo "=========================================="
echo ""
echo "快速查看:"
echo "  grep -E 'acc|Task' ${RESULTS_DIR}/${MODEL_SHORT}.log | tail -20"
