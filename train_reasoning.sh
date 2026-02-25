#!/bin/bash
# ============================================================================
# RADLADS Reasoning 训练脚本 (支持 RWKV6 + RWKV7)
#
# 数据集推荐 (按优先级):
#   1. open-r1/OpenR1-Math-220k      — 220k 数学推理 (R1 traces, ~800k 推理链)
#   2. open-thoughts/OpenThoughts-114k — 114k 多领域推理 (数学/科学/代码/谜题)
#   3. AI-MO/NuminaMath-CoT           — 860k+ 数学竞赛 CoT
#   4. bespokelabs/Bespoke-Stratos-17k — 17k 高质量多样推理
#
# 用法:
#   bash train_reasoning.sh all 20M v6-model2       # RWKV6 Model 2 + 20M
#   bash train_reasoning.sh all 20M v6-model3       # RWKV6 Model 3 + 20M
#   bash train_reasoning.sh all 20M v7-model2       # RWKV7 Model 2 + 20M
#   bash train_reasoning.sh all 20M v7-model3       # RWKV7 Model 3 + 20M
#   bash train_reasoning.sh all 200M v7-model3      # RWKV7 Model 3 + 200M
#
#   # 环境变量方式:
#   MODEL_CKPT=./ckpt/xxx.pth bash train_reasoning.sh all 20M
# ============================================================================
set -e

# === 默认配置 (可通过环境变量覆盖) ===
DATASET="${DATASET:-open-r1/OpenR1-Math-220k}"
CTX_LEN="${CTX_LEN:-4096}"
MAX_TOKENS="${MAX_TOKENS:-20000000}"
MODEL_CKPT="${MODEL_CKPT:-./ckpt/L28-D3584-qwen2-rwkv6-3.pth}"
NUM_DEVICES="${NUM_DEVICES:-2}"
MICRO_BSZ="${MICRO_BSZ:-1}"
PRECISION="${PRECISION:-bf16}"
STRATEGY="${STRATEGY:-deepspeed_stage_2}"
OPTIMIZER="${OPTIMIZER:-adamw}"

# === 解析快捷参数 ===
for arg in "${@:2}"; do
    case "$arg" in
        *[Mm])  # e.g. 20M, 200m
            num="${arg%[Mm]}"
            MAX_TOKENS=$((num * 1000000))
            ;;
        v6-model2|model2)
            MODEL_CKPT="./ckpt/L28-D3584-qwen2-rwkv6-2.pth"
            ;;
        v6-model3|model3)
            MODEL_CKPT="./ckpt/L28-D3584-qwen2-rwkv6-3.pth"
            ;;
        v7-model2)
            MODEL_CKPT="./ckpt/L28-D3584-qwen2-rwkv7-2.pth"
            ;;
        v7-model3)
            MODEL_CKPT="./ckpt/L28-D3584-qwen2-rwkv7-3.pth"
            ;;
    esac
done

# === 自动检测 RWKV6 vs RWKV7 ===
if echo "$MODEL_CKPT" | grep -q "rwkv7"; then
    ARCH="rwkv7"
    ARCH_CONFIG="configs/qwerky7.yaml"
    ATTN_TYPE="rwkv7_fla_chunk"
else
    ARCH="rwkv6"
    ARCH_CONFIG="configs/qwerky6.yaml"
    ATTN_TYPE="gla"
fi

# === 自动推导命名 ===
DATASET_BASENAME=$(basename "$DATASET")

# 数据量标签: 20000000 -> 20M
if [ "$MAX_TOKENS" -ge 1000000000 ]; then
    TOKEN_LABEL="$((MAX_TOKENS / 1000000000))B"
elif [ "$MAX_TOKENS" -ge 1000000 ]; then
    TOKEN_LABEL="$((MAX_TOKENS / 1000000))M"
else
    TOKEN_LABEL="${MAX_TOKENS}"
fi

# 模型标签: ./ckpt/L28-D3584-qwen2-rwkv7-2.pth -> rwkv7-2
MODEL_BASENAME=$(basename "$MODEL_CKPT" .pth)
MODEL_TAG=$(echo "$MODEL_BASENAME" | grep -oP 'rwkv[67]-\d+' || echo "$MODEL_BASENAME")

# 数据路径带数据量后缀: data/OpenR1-Math-220k-20M
DATA_PREFIX="data/${DATASET_BASENAME}-${TOKEN_LABEL}"
PARAMS_FILE="${DATA_PREFIX}_params.txt"

# 训练输出后缀: reasoning-rwkv7-2-OpenR1-Math-220k-20M
PROJ_SUFFIX="reasoning-${MODEL_TAG}-${DATASET_BASENAME}-${TOKEN_LABEL}"

echo "============================================"
echo " Config:"
echo "   Arch:       $ARCH ($ARCH_CONFIG)"
echo "   Attn type:  $ATTN_TYPE"
echo "   Dataset:    $DATASET"
echo "   Tokens:     $MAX_TOKENS ($TOKEN_LABEL)"
echo "   Model:      $MODEL_CKPT ($MODEL_TAG)"
echo "   Data path:  ${DATA_PREFIX}.bin"
echo "   Proj suffix: $PROJ_SUFFIX"
echo "   Devices:    $NUM_DEVICES"
echo "============================================"

# ============================================================================
# Step 1: 准备数据
# ============================================================================
prepare_data() {
    echo ""
    echo "=========================================="
    echo " Step 1: Preparing reasoning data"
    echo " Dataset: $DATASET"
    echo " Tokens:  $MAX_TOKENS ($TOKEN_LABEL)"
    echo " Output:  ${DATA_PREFIX}.bin"
    echo "=========================================="

    # 如果数据已存在，跳过
    if [ -f "$PARAMS_FILE" ] && [ -f "${DATA_PREFIX}.bin" ]; then
        echo "[SKIP] Data already exists at ${DATA_PREFIX}.bin"
        cat "$PARAMS_FILE"
        return 0
    fi

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
    echo " Arch:    $ARCH | Attn: $ATTN_TYPE"
    echo " Model:   $MODEL_CKPT"
    echo " Data:    ${DATA_PREFIX}"
    echo " Suffix:  $PROJ_SUFFIX"
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

    # RWKV6 Model 2 需要指定旧的 lora ranks
    LORA_ARGS=""
    if echo "$MODEL_CKPT" | grep -q "rwkv6-2"; then
        LORA_ARGS="--model.lora_rank_tokenshift 32 --model.lora_rank_decay 64"
        echo " lora_ranks     = tokenshift=32, decay=64 (rwkv6-model2)"
    fi

    RWKV_TORCH_COMPILE=0 RWKV_JIT_ON=0 python train.py \
        -c configs/qwen7b.yaml \
        -c "$ARCH_CONFIG" \
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
        --train.proj_suffix "$PROJ_SUFFIX" \
        --model.attention_type "$ATTN_TYPE" \
        $LORA_ARGS
}

# ============================================================================
# 执行
# ============================================================================
case "${1:-help}" in
    prep)    prepare_data ;;
    train)   train_model ;;
    all)     prepare_data && train_model ;;
    *)
        echo "RADLADS Reasoning 训练脚本 (RWKV6 + RWKV7)"
        echo ""
        echo "用法:"
        echo "  bash train_reasoning.sh prep    — 下载并预处理数据"
        echo "  bash train_reasoning.sh train   — 开始训练"
        echo "  bash train_reasoning.sh all     — 一键执行"
        echo ""
        echo "快捷参数 (在 prep/train/all 后面追加):"
        echo "  20M / 200M              — 设置数据量"
        echo "  v6-model2 / v6-model3   — RWKV6 模型"
        echo "  v7-model2 / v7-model3   — RWKV7 模型"
        echo ""
        echo "示例:"
        echo "  bash train_reasoning.sh all 20M v6-model2"
        echo "  bash train_reasoning.sh all 20M v7-model3"
        echo "  bash train_reasoning.sh all 200M v7-model2"
        echo ""
        echo "当前配置: ${NUM_DEVICES}卡, $STRATEGY, $OPTIMIZER, ctx${CTX_LEN}"
        ;;
esac
