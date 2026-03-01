#!/bin/bash
# ============================================================================
# RADLADS 综合测评脚本 v2 (含 Reasoning 任务, 双卡并行)
#
# 用法:
#   bash eval_reasoning.sh <model> [suite]                    # 单模型
#   bash eval_reasoning.sh all [suite]                        # 全部模型 (双卡并行)
#
# Suite:
#   base       — 基础 benchmark
#   reasoning  — 推理 benchmark (含 gsm8k)
#   all        — 全部 (默认)
#
# 示例:
#   bash eval_reasoning.sh v7-model3                          # 全部任务
#   bash eval_reasoning.sh v7-model3-200M reasoning           # 只测推理
#   bash eval_reasoning.sh all                                # 双卡并行测全部
#   bash eval_reasoning.sh all reasoning                      # 双卡并行只测推理
# ============================================================================
set -e

# === 配置 ===
PRECISION="${PRECISION:-16}"
BSZ="${BSZ:-8}"                  # loglikelihood 任务 batch size
BSZ_GEN="${BSZ_GEN:-1}"         # 生成类任务 batch size
CTX_LEN="${CTX_LEN:-4096}"
RESULTS_DIR="eval_results"
NUM_GPUS=2                      # 可用 GPU 数量

# === 任务集 ===
# 基础 benchmark (loglikelihood, 可以大 batch)
TASKS_BASE="lambada_openai,arc_easy,arc_challenge,hellaswag,winogrande,piqa,openbookqa,boolq"
# 推理 benchmark (loglikelihood)
TASKS_REASONING_MC="sciq,truthfulqa_mc2"
# 推理 benchmark (generation)
TASKS_REASONING_GEN="gsm8k"

# === 模型注册表 ===
declare -A MODEL_PATHS
# 基线模型
MODEL_PATHS[v6-model2]="./ckpt/L28-D3584-qwen2-rwkv6-2.pth"
MODEL_PATHS[v6-model3]="./ckpt/L28-D3584-qwen2-rwkv6-3.pth"
MODEL_PATHS[v7-model2]="./ckpt/L28-D3584-qwen2-rwkv7-2.pth"
MODEL_PATHS[v7-model3]="./ckpt/L28-D3584-qwen2-rwkv7-3.pth"
# 训练后模型 (20M)
MODEL_PATHS[v6-model2-20M]="out/L28-D3584-qwerky6_qwen2-reasoning-rwkv6-2-OpenR1-Math-220k-20M/rwkv-final.pth"
MODEL_PATHS[v6-model3-20M]="out/L28-D3584-qwerky6_qwen2-reasoning-rwkv6-3-OpenR1-Math-220k-20M/rwkv-final.pth"
MODEL_PATHS[v7-model2-20M]="out/L28-D3584-qwerky7_qwen2-reasoning-rwkv7-2-OpenR1-Math-220k-20M/rwkv-final.pth"
MODEL_PATHS[v7-model3-20M]="out/L28-D3584-qwerky7_qwen2-reasoning-rwkv7-3-OpenR1-Math-220k-20M/rwkv-final.pth"
# 训练后模型 (200M)
MODEL_PATHS[v6-model2-200M]="out/L28-D3584-qwerky6_qwen2-reasoning-rwkv6-2-OpenR1-Math-220k-200M/rwkv-final.pth"
MODEL_PATHS[v6-model3-200M]="out/L28-D3584-qwerky6_qwen2-reasoning-rwkv6-3-OpenR1-Math-220k-200M/rwkv-final.pth"
MODEL_PATHS[v7-model2-200M]="out/L28-D3584-qwerky7_qwen2-reasoning-rwkv7-2-OpenR1-Math-220k-200M/rwkv-final.pth"
MODEL_PATHS[v7-model3-200M]="out/L28-D3584-qwerky7_qwen2-reasoning-rwkv7-3-OpenR1-Math-220k-200M/rwkv-final.pth"

# 测评顺序
ALL_KEYS=(
    v6-model2 v6-model3 v7-model2 v7-model3
    v6-model2-20M v6-model3-20M v7-model2-20M v7-model3-20M
    v6-model2-200M v6-model3-200M v7-model2-200M v7-model3-200M
)

# === 解析参数 ===
MODEL_KEY=""
SUITE="all"

for arg in "$@"; do
    if [ -z "$MODEL_KEY" ]; then
        MODEL_KEY="$arg"
    else
        SUITE="$arg"
    fi
done

if [ -z "$MODEL_KEY" ]; then
    echo "RADLADS 综合测评脚本 v2 (双卡并行)"
    echo ""
    echo "用法: bash eval_reasoning.sh <model> [suite]"
    echo ""
    echo "模型快捷名:"
    echo "  基线:   v6-model2, v6-model3, v7-model2, v7-model3"
    echo "  20M:    v6-model2-20M, v6-model3-20M, v7-model2-20M, v7-model3-20M"
    echo "  200M:   v6-model2-200M, v6-model3-200M, v7-model2-200M, v7-model3-200M"
    echo "  全部:   all (双卡并行)"
    echo ""
    echo "Suite: base | reasoning | all (默认)"
    echo ""
    echo "  base      — $TASKS_BASE"
    echo "  reasoning — $TASKS_REASONING_MC + $TASKS_REASONING_GEN"
    echo ""
    echo "示例:"
    echo "  bash eval_reasoning.sh v7-model3-200M reasoning"
    echo "  bash eval_reasoning.sh all"
    exit 0
fi

# === 辅助函数 ===
detect_arch() {
    if echo "$1" | grep -q "rwkv7\|qwerky7"; then echo "rwkv7"; else echo "rwkv6"; fi
}

get_arch_config() {
    if [ "$1" = "rwkv7" ]; then echo "configs/qwerky7.yaml"; else echo "configs/qwerky6.yaml"; fi
}

get_attn_type() {
    if [ "$1" = "rwkv7" ]; then echo "rwkv7_fla_fused_recurrent"; else echo "gla"; fi
}

get_lora_args() {
    if echo "$1" | grep -q "rwkv6-2"; then
        echo "--model.lora_rank_tokenshift 32 --model.lora_rank_decay 64"
    fi
}

resolve_path() {
    local path="$1"
    if [ -f "$path" ]; then echo "$path"; return; fi
    local dir=$(dirname "$path")
    if [ -f "${dir}/rwkv-0.pth" ]; then echo "${dir}/rwkv-0.pth"; return; fi
    echo ""
}

# === 单模型测评 (指定 GPU) ===
eval_one_model() {
    local model_path="$1"
    local model_name="$2"
    local gpu_id="${3:-0}"

    model_path=$(resolve_path "$model_path")
    if [ -z "$model_path" ]; then
        echo "[SKIP] $model_name — 文件不存在"
        return 0
    fi

    local arch=$(detect_arch "$model_path")
    local arch_config=$(get_arch_config "$arch")
    local attn_type=$(get_attn_type "$arch")
    local lora_args=$(get_lora_args "$model_path")

    mkdir -p "$RESULTS_DIR"

    echo "[GPU $gpu_id] $model_name | $arch | $attn_type"

    # --- loglikelihood 任务 ---
    local tasks=""
    case "$SUITE" in
        base)      tasks="$TASKS_BASE" ;;
        reasoning) tasks="$TASKS_REASONING_MC" ;;
        all)       tasks="${TASKS_BASE},${TASKS_REASONING_MC}" ;;
    esac

    if [ -n "$tasks" ]; then
        echo "[GPU $gpu_id] MC tasks (bsz=$BSZ): $tasks"
        CUDA_VISIBLE_DEVICES=$gpu_id python run_lm_eval.py \
            -c configs/qwen7b.yaml \
            -c "$arch_config" \
            --model.attention_type "$attn_type" \
            --model.ctx_len $CTX_LEN \
            --precision $PRECISION \
            --path "$model_path" \
            --tasks "$tasks" \
            --bsz $BSZ \
            $lora_args \
            2>&1 | tee "${RESULTS_DIR}/${model_name}_mc.log"
    fi

    # --- 生成类任务 ---
    if [ "$SUITE" = "all" ] || [ "$SUITE" = "reasoning" ]; then
        echo "[GPU $gpu_id] Gen tasks (bsz=$BSZ_GEN): $TASKS_REASONING_GEN"
        CUDA_VISIBLE_DEVICES=$gpu_id python run_lm_eval.py \
            -c configs/qwen7b.yaml \
            -c "$arch_config" \
            --model.attention_type "$attn_type" \
            --model.ctx_len $CTX_LEN \
            --precision $PRECISION \
            --path "$model_path" \
            --tasks "$TASKS_REASONING_GEN" \
            --bsz $BSZ_GEN \
            $lora_args \
            2>&1 | tee "${RESULTS_DIR}/${model_name}_gen.log"
    fi

    echo "[GPU $gpu_id][DONE] $model_name"
}

# === 双卡并行批量测评 ===
eval_all_parallel() {
    echo "=========================================="
    echo " 双卡并行批量测评"
    echo " Suite: $SUITE | GPUs: $NUM_GPUS"
    echo "=========================================="

    # 收集有效模型
    local valid_keys=()
    for key in "${ALL_KEYS[@]}"; do
        local path="${MODEL_PATHS[$key]}"
        local resolved=$(resolve_path "$path")
        if [ -n "$resolved" ]; then
            valid_keys+=("$key")
        else
            echo "[SKIP] $key — 文件不存在"
        fi
    done

    echo ""
    echo "有效模型: ${#valid_keys[@]} 个"
    echo ""

    # 双卡并行: 每次分配 2 个模型到 GPU 0 和 GPU 1
    local i=0
    while [ $i -lt ${#valid_keys[@]} ]; do
        local key0="${valid_keys[$i]}"
        local key1="${valid_keys[$((i+1))]:-}"

        if [ -n "$key1" ]; then
            echo ""
            echo ">>> 并行: GPU 0 → $key0 | GPU 1 → $key1"
            echo ""
            eval_one_model "${MODEL_PATHS[$key0]}" "$key0" 0 &
            local pid0=$!
            eval_one_model "${MODEL_PATHS[$key1]}" "$key1" 1 &
            local pid1=$!
            wait $pid0 $pid1
            i=$((i + 2))
        else
            echo ""
            echo ">>> GPU 0 → $key0"
            echo ""
            eval_one_model "${MODEL_PATHS[$key0]}" "$key0" 0
            i=$((i + 1))
        fi
    done

    echo ""
    echo "=========================================="
    echo " 全部测评完成! 结果在 $RESULTS_DIR/"
    echo "=========================================="
    echo ""
    echo "查看日志:"
    echo "  ls $RESULTS_DIR/"
    echo ""
    echo "快速汇总:"
    echo "  grep -h 'acc' $RESULTS_DIR/*_mc.log | head -50"
}

# === 执行 ===
if [ "$MODEL_KEY" = "all" ]; then
    eval_all_parallel
elif [ -n "${MODEL_PATHS[$MODEL_KEY]}" ]; then
    eval_one_model "${MODEL_PATHS[$MODEL_KEY]}" "$MODEL_KEY" 0
else
    model_name=$(basename "$(dirname "$MODEL_KEY")")-$(basename "$MODEL_KEY" .pth)
    eval_one_model "$MODEL_KEY" "$model_name" 0
fi
