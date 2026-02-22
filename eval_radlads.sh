#!/bin/bash
# ============================================================================
# RADLADS 测评脚本
# 用法: bash eval_radlads.sh
# ============================================================================
set -e

# === 配置 ===
CKPT_DIR="./ckpt"
PRECISION=16
BSZ=4
CTX_LEN=4096
TASKS="lambada_openai,arc_easy,arc_challenge,hellaswag,winogrande,piqa,openbookqa"

# === Step 0: 环境安装 ===
setup_env() {
    echo "[SETUP] Installing dependencies..."
    pip install torch lightning flash-linear-attention triton lm_eval \
        safetensors transformers ninja deepspeed --upgrade
}

# === Step 1: 修复 bmm CUBLAS 兼容性问题 ===
apply_fix() {
    if grep -q 'transpose(0, 1)$' models/qwen2.py 2>/dev/null; then
        sed -i 's/\.transpose(0, 1)$/\.transpose(0, 1).contiguous()/' models/qwen2.py
        echo "[FIX] Applied .contiguous() fix to models/qwen2.py"
    else
        echo "[FIX] Fix already applied or not needed"
    fi
}

# === Step 2: 测评模型 2 (lora_rank_tokenshift=32, lora_rank_decay=64) ===
eval_model2() {
    echo ""
    echo "=========================================="
    echo " Evaluating: L28-D3584-qwen2-rwkv6-2"
    echo "=========================================="
    python run_lm_eval.py \
        -c configs/qwen7b.yaml \
        -c configs/qwerky6.yaml \
        --model.attention_type gla \
        --model.lora_rank_tokenshift 32 \
        --model.lora_rank_decay 64 \
        --model.ctx_len $CTX_LEN \
        --precision $PRECISION \
        --path $CKPT_DIR/L28-D3584-qwen2-rwkv6-2.pth \
        --tasks $TASKS \
        --bsz $BSZ
}

# === Step 3: 测评模型 3 (默认 lora ranks: 96) ===
eval_model3() {
    echo ""
    echo "=========================================="
    echo " Evaluating: L28-D3584-qwen2-rwkv6-3"
    echo "=========================================="
    python run_lm_eval.py \
        -c configs/qwen7b.yaml \
        -c configs/qwerky6.yaml \
        --model.attention_type gla \
        --model.ctx_len $CTX_LEN \
        --precision $PRECISION \
        --path $CKPT_DIR/L28-D3584-qwen2-rwkv6-3.pth \
        --tasks $TASKS \
        --bsz $BSZ
}

# === 执行 ===
case "${1:-all}" in
    setup)   setup_env ;;
    fix)     apply_fix ;;
    model2)  apply_fix && eval_model2 ;;
    model3)  apply_fix && eval_model3 ;;
    all)     apply_fix && eval_model2 && eval_model3 ;;
    *)       echo "Usage: bash eval_radlads.sh [setup|fix|model2|model3|all]" ;;
esac
