# Qwen2.5-7B-Instruct 从头测试 + 同口径测评参考命令

> 这个指南基于 README 中的 Qwen2.5-7B-Instruct 示例流程，按可直接复制执行的顺序整理，并补充「baseline 与蒸馏模型同口径 lm_eval」命令。

## 0) 环境准备

```bash
pip install lightning torch flash-linear-attention triton deepspeed wandb ninja --upgrade
```

## 1) 下载数据（可选，但训练通常需要）

```bash
mkdir -p data
wget --continue -O data/dclm-10B.idx "https://huggingface.co/datasets/recursal/DCLM-10B-Qwen2-binidx/resolve/main/dclm-10B.idx?download=true"
wget --continue -O data/dclm-10B.bin "https://huggingface.co/datasets/recursal/DCLM-10B-Qwen2-binidx/resolve/main/dclm-10B.bin?download=true"
```

## 2) 下载 Qwen2.5-7B-Instruct

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct
```

## 3) 转换为 PTH（baseline 权重）

```bash
python3 convert_hf_to_pth.py YOUR_CACHED_HF_QWEN_MODEL_LOCATION out/Qwen2.5-7B-Instruct.pth
```

## 4) 训练流程

### Step 0（先跑这个，生成 Step 1 的输入 checkpoint）

```bash
RWKV_TORCH_COMPILE=0 RWKV_JIT_ON=0 \
python3 train.py \
  -c configs/qwen7b.yaml \
  -c configs/qwerky7.yaml \
  -c configs/distill1.yaml \
  --train.load_model out/Qwen2.5-7B-Instruct.pth
```

### Step 1（README 里的重点步骤）

```bash
RWKV_TORCH_COMPILE=0 RWKV_JIT_ON=0 \
python3 train.py \
  -c configs/qwen7b.yaml \
  -c configs/qwerky7.yaml \
  -c configs/qwen7binstructteacher.yaml \
  -c configs/distill2.yaml \
  --train.load_model out/L28-D3584-qwerky7_qwen2-1/rwkv-final.pth
```

### Step 2

```bash
RWKV_TORCH_COMPILE=0 RWKV_JIT_ON=0 \
python3 train.py \
  -c configs/qwen7b.yaml \
  -c configs/qwerky7.yaml \
  -c configs/qwen7binstructteacher.yaml \
  -c configs/distill3.yaml \
  --train.load_model out/L28-D3584-qwerky7_qwen2-2/rwkv-final.pth
```

## 5) 同口径 lm_eval 测评（你这次最关心）

> 关键原则：
> - **baseline (`out/Qwen2.5-7B-Instruct.pth`) 不要叠加 `qwerky6/qwerky7` 配置**，只用 `qwen7b.yaml`，并显式加 `--model.attention_type sdpa`。
> - 蒸馏后的 RWKV6/RWKV7 checkpoint，再叠加对应 `qwerky6.yaml` 或 `qwerky7.yaml` + 对应 attention_type。

### 5.1 baseline（原始 Qwen2.5-7B-Instruct 转换后的 pth）

```bash
python run_lm_eval.py \
  -c configs/qwen7b.yaml \
  --model.ctx_len 4096 \
  --model.attention_type sdpa \
  --precision 16 \
  --path out/Qwen2.5-7B-Instruct.pth \
  --tasks lambada_openai,arc_easy,arc_challenge,hellaswag,winogrande,piqa,openbookqa \
  --bsz 4
```

### 5.2 RWKV6 模型（和你之前命令一致）

```bash
python run_lm_eval.py \
  -c configs/qwen7b.yaml \
  -c configs/qwerky6.yaml \
  --model.attention_type gla \
  --model.ctx_len 4096 \
  --precision 16 \
  --path out/L28-D3584-qwerky6_qwen2-reasoning-rwkv6-3-OpenR1-Math-220k-200M/rwkv-0.pth \
  --tasks lambada_openai,arc_easy,arc_challenge,hellaswag,winogrande,piqa,openbookqa \
  --bsz 4
```

### 5.3 RWKV7 模型（和你之前命令一致）

```bash
python run_lm_eval.py \
  -c configs/qwen7b.yaml \
  -c configs/qwerky7.yaml \
  --model.attention_type rwkv7_fla_fused_recurrent \
  --model.ctx_len 4096 \
  --precision 16 \
  --path ./ckpt/L28-D3584-qwen2-rwkv7-2.pth \
  --tasks lambada_openai,arc_easy,arc_challenge,hellaswag,winogrande,piqa,openbookqa \
  --bsz 4
```

## 6) 转换回 safetensors（可选）

```bash
python3 convert_to_safetensors.py \
  out/L28-D3584-qwerky7_qwen2-3/rwkv-final.pth \
  RADRWKV7Qwen2.5-7B/model.safetensors
```

## 常见注意事项

- `Step 1` 依赖 `Step 0` 的输出；`Step 2` 依赖 `Step 1` 的输出。
- 若在同一输出目录重复运行，项目会从已有编号 checkpoint 继续。
- 建议为每次实验使用独立输出目录，便于对比结果和复现实验。
- 做横向对比时，尽量固定同一组 `tasks / ctx_len / precision / bsz`，只改模型路径与架构配置。
- 如果你遇到 `FileNotFoundError: rwkv_cuda_wind/backstepping_longhead.cu`，说明当前代码仓里的 longhead CUDA 源文件缺失；可改用 `configs/qwerky7.yaml` 默认的 `rwkv7_wind_backstepping_bighead`（本仓库已可用）。
