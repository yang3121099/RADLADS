#!/bin/bash
# ============================================================================
# RADLADS Reasoning 训练脚本
#
# 数据集推荐 (按优先级):
#   1. open-r1/OpenR1-Math-220k      — 220k 数学推理 (R1 traces, ~800k 推理链)
#   2. open-thoughts/OpenThoughts-114k — 114k 多领域推理 (数学/科学/代码/谜题)
#   3. AI-MO/NuminaMath-CoT           — 860k+ 数学竞赛 CoT
#   4. bespokelabs/Bespoke-Stratos-17k — 17k 高质量多样推理
#
# 用法:
#   bash train_reasoning.sh prep      # Step 1: 准备数据
#   bash train_reasoning.sh train     # Step 2: 开始训练
#   bash train_reasoning.sh all       # 一键执行全部
# ============================================================================
set -e

# === 配置 (按需修改) ===
DATASET="open-r1/OpenR1-Math-220k"     # HuggingFace 数据集
CTX_LEN=4096                            # 上下文长度 (4卡可以用 4096)
MAX_TOKENS=200000000                    # 最大处理 token 数 (200M)
MODEL_CKPT="./ckpt/L28-D3584-qwen2-rwkv6-3.pth"  # 基础模型 checkpoint
NUM_DEVICES=4                           # GPU 数量
MICRO_BSZ=2                             # 单卡 batch size (4卡140GB足够)
PRECISION="bf16"                        # bf16 / 16 / 32
STRATEGY="deepspeed_stage_2"            # ZeRO-2: 优化器状态分片到4卡
OPTIMIZER="adamw"                       # 4卡够用 adamw 全精度

# === 自动推导 ===
DATASET_BASENAME=$(basename $DATASET)
DATA_PREFIX="data/${DATASET_BASENAME}"
PARAMS_FILE="${DATA_PREFIX}_params.txt"

# ============================================================================
# Step 1: 准备数据
# ============================================================================
prepare_data() {
    echo "=========================================="
    echo " Step 1: Preparing reasoning data"
    echo " Dataset: $DATASET"
    echo "=========================================="

    python prepare_reasoning_data.py \
        --dataset "$DATASET" \
        --ctxlen $CTX_LEN \
        --max_tokens $MAX_TOKENS \
        --tokenizer "Qwen/Qwen2.5-7B-Instruct" \
        --out "$DATA_PREFIX"

    if [ ! -f "$PARAMS_FILE" ]; then
        echo "[ERROR] Data preparation failed - params file not found"
        exit 1
    fi

    echo ""
    echo "[OK] Data ready at ${DATA_PREFIX}.bin / .idx"
    cat "$PARAMS_FILE"
}

# ============================================================================
# Step 2: 训练
# ============================================================================
train_model() {
    echo ""
    echo "=========================================="
    echo " Step 2: Training (${NUM_DEVICES}x GPU)"
    echo " Model: $MODEL_CKPT"
    echo " Data:  ${DATA_PREFIX}"
    echo " Strategy: $STRATEGY"
    echo "=========================================="

    if [ ! -f "$PARAMS_FILE" ]; then
        echo "[ERROR] Run 'bash train_reasoning.sh prep' first"
        exit 1
    fi

    # 读取数据参数
    source "$PARAMS_FILE"
    echo " my_exit_tokens = $my_exit_tokens"
    echo " magic_prime    = $magic_prime"
    echo " ctx_len        = $ctx_len"

    RWKV_TORCH_COMPILE=0 RWKV_JIT_ON=0 python train.py \
        -c configs/qwen7b.yaml \
        -c configs/qwerky6.yaml \
        -c configs/finetune_reasoning.yaml \
        --train.load_model "$MODEL_CKPT" \
        --train.data_file "$data_file" \
        --train.my_exit_tokens $my_exit_tokens \
        --train.magic_prime $magic_prime \
        --model.ctx_len $ctx_len \
        --train.devices $NUM_DEVICES \
        --train.micro_bsz $MICRO_BSZ \
        --train.precision $PRECISION \
        --train.strategy $STRATEGY \
        --train.optimizer $OPTIMIZER \
        --model.attention_type gla
}

# ============================================================================
# 执行
# ============================================================================
case "${1:-help}" in
    prep)    prepare_data ;;
    train)   train_model ;;
    all)     prepare_data && train_model ;;
    *)
        echo "RADLADS Reasoning 训练脚本 (多卡版)"
        echo ""
        echo "用法:"
        echo "  bash train_reasoning.sh prep    — 下载并预处理数据"
        echo "  bash train_reasoning.sh train   — 开始训练"
        echo "  bash train_reasoning.sh all     — 一键执行"
        echo ""
        echo "当前配置: ${NUM_DEVICES}卡, $STRATEGY, $OPTIMIZER, ctx${CTX_LEN}"
        echo "可选: 修改脚本头部的配置项来切换数据集或模型"
        ;;
esac
