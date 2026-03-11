#!/bin/bash
# ============================================================================
# RADLADS 继续训练脚本 - 从 step2-ckpt1 恢复训练
#
# 用法:
#   bash run_continue_train.sh setup          # 下载 checkpoint + 准备数据
#   bash run_continue_train.sh prepare_data   # 仅准备数据
#   bash run_continue_train.sh train          # 开始训练 (默认用 OpenR1-Math)
#   bash run_continue_train.sh train_chimera  # 用 CHIMERA 数据训练
#   bash run_continue_train.sh eval <path>    # 测评指定 checkpoint (自动跳过已完成)
#   bash run_continue_train.sh eval_all       # 测评所有 checkpoint (自动跳过已完成)
#   bash run_continue_train.sh summary        # 输出汇总 markdown 表格
# ============================================================================
set -e

CKPT_URL="https://huggingface.co/yang31210999/L28-D3584-qwerky7_qwen2-4_BOA/resolve/main/rwkv-1.pth"
CKPT_DIR="out/L28-D3584-qwerky7_qwen2-4_BOA"
CKPT_PATH="${CKPT_DIR}/rwkv-1.pth"
PROJ_DIR="out/L28-D3584-qwerky7_qwen2-5_continue"

EVAL_BSZ=4
EVAL_WORKERS_PER_GPU=2  # 每张卡同时跑几组 eval

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
            --ctxlen 4096 \
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
            --ctxlen 4096 \
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
        --train.data_file data/OpenR1-Math-220k \
        --train.magic_prime 98939
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
        --train.data_file data/CHIMERA \
        --train.magic_prime 53939
}

# === Step 3: 测评 (使用 eval_manager.py) ===
eval_model() {
    local model_path="${1}"
    if [ -z "${model_path}" ]; then
        echo "Usage: bash run_continue_train.sh eval <checkpoint_path>"
        exit 1
    fi
    python eval_manager.py eval --path "${model_path}" --bsz ${EVAL_BSZ}
}

# === Step 4: 测评所有 checkpoint (单卡串行) ===
eval_all() {
    local dir="${2:-${PROJ_DIR}}"
    python eval_manager.py eval_all --dir "${dir}" --bsz ${EVAL_BSZ}
}

# === Step 4b: 双卡并行测评所有 checkpoint (推荐) ===
eval_all_parallel() {
    local dir="${2:-${PROJ_DIR}}"
    python eval_manager.py eval_all_parallel --dir "${dir}" --bsz ${EVAL_BSZ} --gpu0 0 --gpu1 1 --workers-per-gpu ${EVAL_WORKERS_PER_GPU}
}

# === Step 5: 输出汇总表格 ===
summary() {
    python eval_manager.py summary
}

# === 执行 ===
case "${1:-help}" in
    setup)              download_ckpt && prepare_data ;;
    download)           download_ckpt ;;
    prepare_data)       prepare_data ;;
    train)              train ;;
    train_chimera)      train_chimera ;;
    eval)               eval_model "${2}" ;;
    eval_all)           eval_all "$@" ;;
    eval_all_parallel)  eval_all_parallel "$@" ;;
    summary)            summary ;;
    *)
        echo "用法: bash run_continue_train.sh <command>"
        echo ""
        echo "Commands:"
        echo "  setup              - 下载 checkpoint + 准备数据"
        echo "  download           - 仅下载 checkpoint"
        echo "  prepare_data       - 仅准备数据 (OpenR1-Math + CHIMERA)"
        echo "  train              - 用 OpenR1-Math 数据训练"
        echo "  train_chimera      - 用 CHIMERA 数据训练"
        echo "  eval <path>        - 测评单个 checkpoint (已完成的自动跳过)"
        echo "  eval_all [dir]     - 单卡串行测评所有 checkpoint"
        echo "  eval_all_parallel  - 双卡并行测评所有 checkpoint (推荐)"
        echo "  summary            - 输出汇总 markdown 表格"
        ;;
esac
