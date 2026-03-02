#!/bin/bash
# ============================================================================
# Qwen 基线测评脚本 (依次测评, 自动清理缓存)
#
# 依次测评以下 4 个模型:
#   1. Qwen/Qwen2-7B
#   2. Qwen/Qwen2-7B-Instruct
#   3. Qwen/Qwen2.5-7B
#   4. Qwen/Qwen2.5-7B-Instruct
#
# 每个模型测完后自动删除 HF 缓存以节省磁盘空间。
#
# 用法:
#   bash eval_qwen_baseline.sh           # 测评全部 4 个模型
#   bash eval_qwen_baseline.sh 2         # 从第 2 个模型开始 (跳过已完成的)
# ============================================================================
set -e

# === 配置 ===
BSZ="${BSZ:-auto}"
RESULTS_DIR="eval_results"
GPU="${GPU:-0}"
HF_CACHE="/workspace/.hf_home/hub"

# === 任务 (与 eval_reasoning.sh 的 TASKS_BASE 保持一致) ===
TASKS="lambada_openai,arc_easy,arc_challenge,hellaswag,winogrande,piqa,openbookqa,boolq"

# === 待测模型列表 ===
MODELS=(
    "Qwen/Qwen2-7B"
    "Qwen/Qwen2-7B-Instruct"
    "Qwen/Qwen2.5-7B"
    "Qwen/Qwen2.5-7B-Instruct"
)

START_FROM="${1:-1}"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo " Qwen Baseline Sequential Evaluation"
echo " Models: ${#MODELS[@]}"
echo " Tasks:  $TASKS"
echo " GPU:    $GPU"
echo " Cache:  $HF_CACHE (auto-cleanup)"
echo "=========================================="
echo ""

for i in "${!MODELS[@]}"; do
    IDX=$((i + 1))
    MODEL="${MODELS[$i]}"
    MODEL_SHORT=$(echo "$MODEL" | sed 's|/|-|g')

    # 跳过已完成的
    if [ "$IDX" -lt "$START_FROM" ]; then
        echo "[$IDX/${#MODELS[@]}] SKIP $MODEL (start_from=$START_FROM)"
        continue
    fi

    echo "=========================================="
    echo " [$IDX/${#MODELS[@]}] $MODEL"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=$GPU lm_eval \
        --model hf \
        --model_args "pretrained=$MODEL,torch_dtype=float16,trust_remote_code=True" \
        --tasks $TASKS \
        --batch_size $BSZ \
        --device cuda \
        --output_path "${RESULTS_DIR}/${MODEL_SHORT}" \
        2>&1 | tee "${RESULTS_DIR}/${MODEL_SHORT}.log"

    echo ""
    echo "[$IDX/${#MODELS[@]}] $MODEL 测评完成"
    echo ""

    # 清理 HF 缓存
    if [ -d "$HF_CACHE" ]; then
        echo "清理 HF 缓存: $HF_CACHE"
        rm -rf "$HF_CACHE"
        mkdir -p "$HF_CACHE"
        echo "缓存已清理"
    fi
    echo ""
done

echo "=========================================="
echo " 全部测评完成!"
echo " 结果目录: $RESULTS_DIR/"
echo "=========================================="
echo ""
echo "查看结果:"
for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT=$(echo "$MODEL" | sed 's|/|-|g')
    echo "  cat ${RESULTS_DIR}/${MODEL_SHORT}.log"
done
