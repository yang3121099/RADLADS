#!/bin/bash
# ============================================================================
# RADLADS 综合测评脚本 v2 (含 Reasoning 任务)
#
# 用法:
#   bash eval_reasoning.sh <model_path>                       # 全部任务
#   bash eval_reasoning.sh <model_path> --suite base          # 基础任务
#   bash eval_reasoning.sh <model_path> --suite reasoning     # 推理任务
#   bash eval_reasoning.sh <model_path> --suite all           # 全部任务
#
# 快捷方式:
#   bash eval_reasoning.sh v6-model2                          # 基线模型
#   bash eval_reasoning.sh v7-model3-200M                     # 训练后模型
#   bash eval_reasoning.sh all                                # 测评全部模型
#
# 结果保存在 eval_results/ 目录
# ============================================================================
set -e

# === 配置 ===
PRECISION="${PRECISION:-16}"
BSZ="${BSZ:-4}"
BSZ_GEN="${BSZ_GEN:-1}"          # 生成类任务 (gsm8k) 的 batch size
CTX_LEN="${CTX_LEN:-4096}"
RESULTS_DIR="eval_results"

# === 任务集 ===
# 基础 benchmark (loglikelihood, 可以 batch)
TASKS_BASE="lambada_openai,arc_easy,arc_challenge,hellaswag,winogrande,piqa,openbookqa,boolq"
# 推理 benchmark (loglikelihood)
TASKS_REASONING_MC="mathqa,logiqa2"
# 推理 benchmark (generation, 需要 bsz=1)
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

# === 解析参数 ===
MODEL_KEY=""
MODEL_PTH=""
SUITE="all"

for arg in "$@"; do
    case "$arg" in
        --suite)    shift; SUITE="$1"; shift ;;
        base|reasoning|all)
            # 如果前面没有 --suite，可能是 suite 参数或 "all" 模型
            if [ "$arg" = "all" ] && [ -z "$MODEL_KEY" ]; then
                MODEL_KEY="all"
            else
                SUITE="$arg"
            fi
            ;;
        *)
            if [ -z "$MODEL_KEY" ]; then
                MODEL_KEY="$arg"
            else
                SUITE="$arg"
            fi
            ;;
    esac
done

if [ -z "$MODEL_KEY" ]; then
    echo "RADLADS 综合测评脚本 v2"
    echo ""
    echo "用法: bash eval_reasoning.sh <model> [suite]"
    echo ""
    echo "模型快捷名:"
    echo "  基线:   v6-model2, v6-model3, v7-model2, v7-model3"
    echo "  20M:    v6-model2-20M, v6-model3-20M, v7-model2-20M, v7-model3-20M"
    echo "  200M:   v6-model2-200M, v6-model3-200M, v7-model2-200M, v7-model3-200M"
    echo "  全部:   all"
    echo "  自定义: 直接传 .pth 路径"
    echo ""
    echo "Suite:"
    echo "  base       — $TASKS_BASE"
    echo "  reasoning  — $TASKS_REASONING_MC + $TASKS_REASONING_GEN"
    echo "  all        — 全部任务 (默认)"
    echo ""
    echo "示例:"
    echo "  bash eval_reasoning.sh v7-model3"
    echo "  bash eval_reasoning.sh v7-model3-200M reasoning"
    echo "  bash eval_reasoning.sh all"
    exit 0
fi

# === 自动检测架构 ===
detect_arch() {
    local path="$1"
    if echo "$path" | grep -q "rwkv7\|qwerky7"; then
        echo "rwkv7"
    else
        echo "rwkv6"
    fi
}

get_arch_config() {
    local arch="$1"
    if [ "$arch" = "rwkv7" ]; then
        echo "configs/qwerky7.yaml"
    else
        echo "configs/qwerky6.yaml"
    fi
}

get_attn_type() {
    local arch="$1"
    if [ "$arch" = "rwkv7" ]; then
        echo "rwkv7_fla_fused_recurrent"
    else
        echo "gla"
    fi
}

get_lora_args() {
    local path="$1"
    # 只有 rwkv6-2 基线需要旧 lora ranks，训练后的模型继承基线的 ranks
    if echo "$path" | grep -q "rwkv6-2"; then
        echo "--model.lora_rank_tokenshift 32 --model.lora_rank_decay 64"
    fi
}

# === 单模型测评函数 ===
eval_one_model() {
    local model_path="$1"
    local model_name="$2"

    if [ ! -f "$model_path" ]; then
        # 检查 rwkv-0.pth 作为备选
        local dir=$(dirname "$model_path")
        if [ -f "${dir}/rwkv-0.pth" ]; then
            model_path="${dir}/rwkv-0.pth"
        else
            echo "[SKIP] $model_name — 文件不存在: $model_path"
            return 0
        fi
    fi

    local arch=$(detect_arch "$model_path")
    local arch_config=$(get_arch_config "$arch")
    local attn_type=$(get_attn_type "$arch")
    local lora_args=$(get_lora_args "$model_path")
    local result_file="${RESULTS_DIR}/${model_name}.json"

    echo ""
    echo "================================================================"
    echo " Model:  $model_name"
    echo " Path:   $model_path"
    echo " Arch:   $arch | Attn: $attn_type"
    echo " Result: $result_file"
    echo "================================================================"

    mkdir -p "$RESULTS_DIR"

    # --- 基础 + MC推理 任务 (可 batch) ---
    if [ "$SUITE" = "all" ] || [ "$SUITE" = "base" ] || [ "$SUITE" = "reasoning" ]; then
        local tasks=""
        case "$SUITE" in
            base)      tasks="$TASKS_BASE" ;;
            reasoning) tasks="$TASKS_REASONING_MC" ;;
            all)       tasks="${TASKS_BASE},${TASKS_REASONING_MC}" ;;
        esac

        if [ -n "$tasks" ]; then
            echo ""
            echo "[EVAL] Loglikelihood tasks (bsz=$BSZ): $tasks"
            python run_lm_eval.py \
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
    fi

    # --- 生成类推理任务 (bsz=1) ---
    if [ "$SUITE" = "all" ] || [ "$SUITE" = "reasoning" ]; then
        echo ""
        echo "[EVAL] Generation tasks (bsz=$BSZ_GEN): $TASKS_REASONING_GEN"
        python run_lm_eval.py \
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

    echo ""
    echo "[DONE] $model_name 测评完成"
}

# === 执行 ===
if [ "$MODEL_KEY" = "all" ]; then
    echo "=========================================="
    echo " 批量测评全部模型"
    echo " Suite: $SUITE"
    echo "=========================================="

    for key in \
        v6-model2 v6-model3 v7-model2 v7-model3 \
        v6-model2-20M v6-model3-20M v7-model2-20M v7-model3-20M \
        v6-model2-200M v6-model3-200M v7-model2-200M v7-model3-200M; do
        path="${MODEL_PATHS[$key]}"
        if [ -n "$path" ]; then
            eval_one_model "$path" "$key"
        fi
    done

    echo ""
    echo "=========================================="
    echo " 全部测评完成! 结果在 $RESULTS_DIR/"
    echo "=========================================="
    echo ""
    echo "查看结果:"
    echo "  ls $RESULTS_DIR/"
    echo "  cat $RESULTS_DIR/*_mc.log"

elif [ -n "${MODEL_PATHS[$MODEL_KEY]}" ]; then
    # 快捷名
    eval_one_model "${MODEL_PATHS[$MODEL_KEY]}" "$MODEL_KEY"
else
    # 直接路径
    model_name=$(basename "$(dirname "$MODEL_KEY")")-$(basename "$MODEL_KEY" .pth)
    eval_one_model "$MODEL_KEY" "$model_name"
fi
