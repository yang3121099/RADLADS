#!/bin/bash
# ============================================================================
# RADLADS 继续训练脚本 - 从 step2-ckpt1 恢复训练
#
# 用法:
#   bash run_continue_train.sh setup          # 下载 checkpoint + 准备数据
#   bash run_continue_train.sh prepare_data   # 仅准备数据
#   bash run_continue_train.sh train          # 开始训练 (默认用 OpenR1-Math)
#   bash run_continue_train.sh train_chimera  # 用 CHIMERA 数据训练
#   bash run_continue_train.sh eval <path>    # 测评指定 checkpoint
#   bash run_continue_train.sh eval_all       # 测评所有已保存 checkpoint
# ============================================================================
set -e

CKPT_URL="https://huggingface.co/yang31210999/L28-D3584-qwerky7_qwen2-4_BOA/resolve/main/rwkv-1.pth"
CKPT_DIR="out/L28-D3584-qwerky7_qwen2-4_BOA"
CKPT_PATH="${CKPT_DIR}/rwkv-1.pth"

EVAL_TASKS="lambada_openai,arc_easy,arc_challenge,hellaswag,winogrande,piqa,openbookqa"
EVAL_BSZ=4
EVAL_PRECISION="bf16"
EVAL_CTX_LEN=4096

# === Step 0: 下载 checkpoint ===
download_ckpt() {
    echo "[DOWNLOAD] Downloading checkpoint..."
    mkdir -p "${CKPT_DIR}"
    if [ -f "${CKPT_PATH}" ]; then
        echo "[DOWNLOAD] Checkpoint already exists: ${CKPT_PATH}"
    else
        wget -O "${CKPT_PATH}" "${CKPT_URL}"
        echo "[DOWNLOAD] Done: ${CKPT_PATH}"
    fi
}

# === Step 1: 准备数据 ===
prepare_data() {
    echo ""
    echo "[DATA] Preparing OpenR1-Math-220k dataset..."
    if [ -f "data/OpenR1-Math-220k.bin" ]; then
        echo "[DATA] OpenR1-Math-220k already prepared, skipping."
    else
        python prepare_reasoning_data.py \
            --dataset open-r1/OpenR1-Math-220k \
            --ctxlen 512 \
            --tokenizer Qwen/Qwen2.5-7B-Instruct \
            --out data/OpenR1-Math-220k \
            --max_tokens 500000000
    fi

    echo ""
    echo "[DATA] Preparing CHIMERA dataset..."
    if [ -f "data/CHIMERA.bin" ]; then
        echo "[DATA] CHIMERA already prepared, skipping."
    else
        python prepare_reasoning_data.py \
            --dataset TianHongZXY/CHIMERA \
            --ctxlen 512 \
            --tokenizer Qwen/Qwen2.5-7B-Instruct \
            --out data/CHIMERA \
            --max_tokens 500000000
    fi
}

# === Step 2: 训练 (OpenR1-Math) ===
train() {
    echo ""
    echo "=========================================="
    echo " Training from step2-ckpt1 with OpenR1-Math"
    echo "=========================================="
    RWKV_TORCH_COMPILE=0 RWKV_JIT_ON=0 python3 train.py \
        -c configs/qwen7b.yaml \
        -c configs/qwerky7.yaml \
        -c configs/continue_train.yaml \
        --model.attention_type rwkv7_fla_chunk \
        --train.load_model "${CKPT_PATH}" \
        --train.data_file data/OpenR1-Math-220k
}

# === Step 2b: 训练 (CHIMERA) ===
train_chimera() {
    echo ""
    echo "=========================================="
    echo " Training from step2-ckpt1 with CHIMERA"
    echo "=========================================="
    RWKV_TORCH_COMPILE=0 RWKV_JIT_ON=0 python3 train.py \
        -c configs/qwen7b.yaml \
        -c configs/qwerky7.yaml \
        -c configs/continue_train.yaml \
        --model.attention_type rwkv7_fla_chunk \
        --train.load_model "${CKPT_PATH}" \
        --train.data_file data/CHIMERA
}

# === Step 3: 测评单个 checkpoint ===
eval_model() {
    local model_path="${1}"
    if [ -z "${model_path}" ]; then
        echo "Usage: bash run_continue_train.sh eval <checkpoint_path>"
        exit 1
    fi
    echo ""
    echo "=========================================="
    echo " Evaluating: ${model_path}"
    echo "=========================================="

    # 基础 benchmarks
    echo "[EVAL] Running standard benchmarks..."
    python run_lm_eval.py \
        -c configs/qwen7b.yaml \
        -c configs/qwerky7.yaml \
        --model.attention_type rwkv7_fla_fused_recurrent \
        --model.ctx_len ${EVAL_CTX_LEN} \
        --precision ${EVAL_PRECISION} \
        --path "${model_path}" \
        --tasks ${EVAL_TASKS} \
        --bsz ${EVAL_BSZ}

    # SuperGPQA (standalone evaluator)
    echo ""
    echo "[EVAL] Running SuperGPQA..."
    python eval_supergpqa.py \
        -c configs/qwen7b.yaml \
        -c configs/qwerky7.yaml \
        --model.attention_type rwkv7_fla_fused_recurrent \
        --model.ctx_len ${EVAL_CTX_LEN} \
        --precision ${EVAL_PRECISION} \
        --path "${model_path}" \
        --bsz ${EVAL_BSZ}
}

# === Step 4: 测评所有已保存的 checkpoint ===
eval_all() {
    local proj_dir="out/L28-D3584-qwerky7_qwen2-5_continue"
    echo "[EVAL] Scanning for checkpoints in ${proj_dir}..."
    for ckpt in "${proj_dir}"/rwkv-*.pth; do
        if [ -f "${ckpt}" ]; then
            eval_model "${ckpt}"
        fi
    done
}

# === 执行 ===
case "${1:-help}" in
    setup)          download_ckpt && prepare_data ;;
    download)       download_ckpt ;;
    prepare_data)   prepare_data ;;
    train)          train ;;
    train_chimera)  train_chimera ;;
    eval)           eval_model "${2}" ;;
    eval_all)       eval_all ;;
    *)
        echo "用法: bash run_continue_train.sh <command>"
        echo ""
        echo "Commands:"
        echo "  setup          - 下载 checkpoint + 准备数据"
        echo "  download       - 仅下载 checkpoint"
        echo "  prepare_data   - 仅准备数据 (OpenR1-Math + CHIMERA)"
        echo "  train          - 用 OpenR1-Math 数据训练"
        echo "  train_chimera  - 用 CHIMERA 数据训练"
        echo "  eval <path>    - 测评单个 checkpoint"
        echo "  eval_all       - 测评 proj 目录下所有 checkpoint"
        ;;
esac
