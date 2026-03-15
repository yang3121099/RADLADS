#!/bin/bash
# ============================================================================
# RADLADS Chatbot SFT 训练 + 测评 一体脚本
#
# 从 step2-ckpt1 (BOA) 继续训练 chatbot SFT 数据
# 双卡训练, 200M tokens, 每 20M 保存一次 checkpoint
# 训练完成后自动测评所有 checkpoint
#
# 推荐数据集:
#   1. SlimOrca        - 518k 高质量 GPT-4 指令 (推荐首选)
#   2. ultrachat_200k  - 200k 多轮对话
#   3. OpenHermes-2.5  - 1M 多源混合指令
#   4. WizardLM_evol_instruct_70k - 70k 进化指令
#   5. CodeFeedback    - 157k 代码指令
#
# 用法:
#   bash run_chatbot_sft.sh setup                    # 下载 ckpt + 准备所有数据
#   bash run_chatbot_sft.sh prepare_data             # 仅准备数据
#   bash run_chatbot_sft.sh train_slimorca           # 用 SlimOrca 训练
#   bash run_chatbot_sft.sh train_ultrachat          # 用 UltraChat 训练
#   bash run_chatbot_sft.sh train_openhermes         # 用 OpenHermes 训练
#   bash run_chatbot_sft.sh train_all                # 依次用所有数据集训练
#   bash run_chatbot_sft.sh eval_all                 # 测评所有已训练的 checkpoint
#   bash run_chatbot_sft.sh eval_baseline            # 测评 Qwen2.5-7B-Instruct baseline
#   bash run_chatbot_sft.sh full                     # 完整流程: setup + train_all + eval_all
# ============================================================================
set -e

# === 配置 ===
CKPT_URL="https://huggingface.co/yang31210999/L28-D3584-qwerky7_qwen2-4_BOA/resolve/main/rwkv-1.pth"
CKPT_DIR="out/L28-D3584-qwerky7_qwen2-4_BOA"
CKPT_PATH="${CKPT_DIR}/rwkv-1.pth"

TOKENIZER="Qwen/Qwen2.5-7B-Instruct"
CTX_LEN=4096
MAX_TOKENS=200000000  # 200M

EVAL_BSZ=1
EVAL_GPU=0  # 测评用的 GPU

# === Step 0: 下载 checkpoint ===
download_ckpt() {
    echo "[DOWNLOAD] Downloading step2-ckpt1 (BOA)..."
    mkdir -p "${CKPT_DIR}"
    if [ -f "${CKPT_PATH}" ]; then
        echo "[DOWNLOAD] Checkpoint already exists: ${CKPT_PATH}"
    else
        wget -O "${CKPT_PATH}" "${CKPT_URL}"
        echo "[DOWNLOAD] Done: ${CKPT_PATH}"
    fi
}

# === Step 1: 准备数据 ===
prepare_one_dataset() {
    local hf_name="$1"
    local out_name="$2"

    echo ""
    echo "[DATA] Preparing ${hf_name}..."
    if [ -f "data/${out_name}.bin" ]; then
        echo "[DATA] ${out_name} already prepared, skipping."
    else
        python prepare_chatbot_data.py \
            --dataset "${hf_name}" \
            --ctxlen ${CTX_LEN} \
            --tokenizer ${TOKENIZER} \
            --out "data/${out_name}" \
            --max_tokens ${MAX_TOKENS}
    fi
}

prepare_data() {
    prepare_one_dataset "Open-Orca/SlimOrca" "SlimOrca"
    prepare_one_dataset "HuggingFaceH4/ultrachat_200k" "ultrachat_200k"
    prepare_one_dataset "teknium/OpenHermes-2.5" "OpenHermes-2.5"
}

# === Step 2: 训练函数 (通用) ===
train_with_dataset() {
    local data_name="$1"
    local magic_prime="$2"
    local suffix="$3"

    echo ""
    echo "=========================================="
    echo " Training chatbot SFT: ${data_name}"
    echo " Suffix: ${suffix}"
    echo " magic_prime: ${magic_prime}"
    echo "=========================================="

    # 从 params.txt 读取 magic_prime (如果没有手动指定)
    if [ "${magic_prime}" == "auto" ]; then
        if [ -f "data/${data_name}_params.txt" ]; then
            magic_prime=$(grep "magic_prime=" "data/${data_name}_params.txt" | cut -d= -f2)
            echo "[TRAIN] Auto-detected magic_prime=${magic_prime} from params file"
        else
            echo "[ERROR] No params file found and magic_prime=auto, please specify manually"
            exit 1
        fi
    fi

    RWKV_TORCH_COMPILE=0 RWKV_JIT_ON=0 python3 train.py \
        -c configs/qwen7b.yaml \
        -c configs/qwerky7.yaml \
        -c configs/chatbot_sft.yaml \
        --model.attention_type rwkv7_fla_chunk \
        --train.load_model "${CKPT_PATH}" \
        --train.data_file "data/${data_name}" \
        --train.magic_prime "${magic_prime}" \
        --train.proj_suffix "6_chatbot_${suffix}" \
        --train.my_exit_tokens ${MAX_TOKENS}
}

train_slimorca() {
    train_with_dataset "SlimOrca" "auto" "slimorca"
}

train_ultrachat() {
    train_with_dataset "ultrachat_200k" "auto" "ultrachat"
}

train_openhermes() {
    train_with_dataset "OpenHermes-2.5" "auto" "openhermes"
}

train_all() {
    train_slimorca
    train_ultrachat
    train_openhermes
}

# === Step 3: 测评 ===
eval_single() {
    local model_path="${1}"
    if [ -z "${model_path}" ]; then
        echo "Usage: bash run_chatbot_sft.sh eval <checkpoint_path>"
        exit 1
    fi
    python eval_chatbot.py eval --path "${model_path}" --bsz ${EVAL_BSZ} --gpu ${EVAL_GPU}
}

eval_all_ckpts() {
    echo ""
    echo "=========================================="
    echo " Evaluating all chatbot SFT checkpoints"
    echo "=========================================="

    # 先评测 baseline (如果还没评过)
    echo "[EVAL] Evaluating baselines..."
    python eval_chatbot.py eval_baseline --model Qwen/Qwen2.5-7B --gpu ${EVAL_GPU} --bsz ${EVAL_BSZ}
    python eval_chatbot.py eval_baseline --model Qwen/Qwen2.5-7B-Instruct --gpu ${EVAL_GPU} --bsz ${EVAL_BSZ}

    # 评测所有数据集的 checkpoint
    for dir in out/L28-D3584-qwerky7_qwen2-6_chatbot_*; do
        if [ -d "${dir}" ]; then
            echo ""
            echo "[EVAL] Evaluating checkpoints in: ${dir}"
            python eval_chatbot.py eval_all --dir "${dir}" --bsz ${EVAL_BSZ} --gpu ${EVAL_GPU}
        fi
    done

    # 也评测 step2-ckpt1 本身
    if [ -f "${CKPT_PATH}" ]; then
        echo ""
        echo "[EVAL] Evaluating base checkpoint: ${CKPT_PATH}"
        python eval_chatbot.py eval --path "${CKPT_PATH}" --bsz ${EVAL_BSZ} --gpu ${EVAL_GPU}
    fi

    python eval_chatbot.py summary
}

eval_baseline() {
    python eval_chatbot.py eval_baseline --model Qwen/Qwen2.5-7B --gpu ${EVAL_GPU} --bsz ${EVAL_BSZ}
    python eval_chatbot.py eval_baseline --model Qwen/Qwen2.5-7B-Instruct --gpu ${EVAL_GPU} --bsz ${EVAL_BSZ}
}

summary() {
    python eval_chatbot.py summary
}

# === 完整流程 ===
full() {
    download_ckpt
    prepare_data
    train_all
    eval_all_ckpts
}

# === 执行 ===
case "${1:-help}" in
    setup)              download_ckpt && prepare_data ;;
    download)           download_ckpt ;;
    prepare_data)       prepare_data ;;
    train_slimorca)     train_slimorca ;;
    train_ultrachat)    train_ultrachat ;;
    train_openhermes)   train_openhermes ;;
    train_all)          train_all ;;
    eval)               eval_single "${2}" ;;
    eval_all)           eval_all_ckpts ;;
    eval_baseline)      eval_baseline ;;
    summary)            summary ;;
    full)               full ;;
    *)
        echo "============================================"
        echo " RADLADS Chatbot SFT Training & Evaluation"
        echo "============================================"
        echo ""
        echo "用法: bash run_chatbot_sft.sh <command>"
        echo ""
        echo "Setup:"
        echo "  setup              - 下载 checkpoint + 准备所有数据"
        echo "  download           - 仅下载 checkpoint"
        echo "  prepare_data       - 准备 SFT 数据 (SlimOrca + UltraChat + OpenHermes)"
        echo ""
        echo "Training:"
        echo "  train_slimorca     - 用 SlimOrca 训练 (推荐首选)"
        echo "  train_ultrachat    - 用 UltraChat-200k 训练"
        echo "  train_openhermes   - 用 OpenHermes-2.5 训练"
        echo "  train_all          - 依次用所有数据集训练"
        echo ""
        echo "Evaluation:"
        echo "  eval <path>        - 测评单个 checkpoint"
        echo "  eval_all           - 测评所有 chatbot SFT checkpoint + baselines"
        echo "  eval_baseline      - 仅测评 Qwen2.5-7B / 7B-Instruct baselines"
        echo "  summary            - 输出汇总 markdown 表格"
        echo ""
        echo "Full Pipeline:"
        echo "  full               - 完整流程: setup + train_all + eval_all"
        ;;
esac
