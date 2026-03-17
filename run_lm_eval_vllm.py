########################################################################################################
# Run lm_eval with vllm or hf backend on a HuggingFace-format model directory.
#
# Prerequisites:
#   pip install lm_eval vllm --upgrade
#
# Usage:
#   # vllm backend (standard Qwen2 models - fastest):
#   python run_lm_eval_vllm.py --model /path/to/hf_model --tasks lambada_openai
#   python run_lm_eval_vllm.py --model /path/to/hf_model --tasks gsm8k,arc_challenge --num_fewshot 5 --bsz auto
#
#   # hf backend (RWKV hybrid models with trust_remote_code):
#   python run_lm_eval_vllm.py --model /path/to/hf_model --tasks lambada_openai --backend hf --bsz 16
#
#   # vllm with tensor parallelism:
#   python run_lm_eval_vllm.py --model /path/to/hf_model --tasks gsm8k --tp 2
#
#   # Conversion + eval in one step (from training checkpoint):
#   python convert_to_hf.py -c configs/qwen7b.yaml --path ckpt.safetensors --out hf_model
#   python run_lm_eval_vllm.py --model hf_model --tasks lambada_openai,gsm8k
#
########################################################################################################

import argparse
import json
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Run lm_eval with vllm/hf backend')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to HuggingFace model directory (output of convert_to_hf.py)')
    parser.add_argument('--tasks', type=str, default='lambada_openai',
                        help='Comma-separated list of tasks (default: lambada_openai)')
    parser.add_argument('--backend', type=str, default='auto', choices=['auto', 'vllm', 'hf'],
                        help='Backend: vllm (fastest, standard models), hf (trust_remote_code models), auto (detect)')
    parser.add_argument('--bsz', type=str, default='auto',
                        help='Batch size (default: auto)')
    parser.add_argument('--num_fewshot', type=int, default=0,
                        help='Number of few-shot examples (default: 0)')
    parser.add_argument('--precision', type=str, default='bfloat16',
                        choices=['float16', 'bfloat16', 'float32', 'auto'],
                        help='Model precision (default: bfloat16)')
    parser.add_argument('--tp', type=int, default=1,
                        help='Tensor parallel size for vllm (default: 1)')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.85,
                        help='GPU memory utilization for vllm (default: 0.85)')
    parser.add_argument('--max_model_len', type=int, default=None,
                        help='Max model length for vllm (default: from config)')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed (default: 1234)')
    parser.add_argument('--limit', type=float, default=None,
                        help='Limit number of examples per task (for debugging)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save results JSON')
    return parser.parse_args()

def detect_backend(model_path: str) -> str:
    """Auto-detect whether to use vllm or hf backend based on model type."""
    config_path = os.path.join(model_path, 'config.json')
    if not os.path.exists(config_path):
        print(f'WARNING: config.json not found in {model_path}, defaulting to vllm')
        return 'vllm'

    with open(config_path, 'r') as f:
        cfg = json.load(f)

    model_type = cfg.get('model_type', '')
    has_auto_map = 'auto_map' in cfg

    if model_type == 'qwen2' and not has_auto_map:
        print(f'Detected standard Qwen2 model -> using vllm backend')
        return 'vllm'
    else:
        print(f'Detected custom model type "{model_type}" (trust_remote_code) -> using hf backend')
        return 'hf'

def main():
    args = parse_args()

    # ---- Auto-detect backend ----
    if args.backend == 'auto':
        backend = detect_backend(args.model)
    else:
        backend = args.backend

    # ---- Import lm_eval ----
    from lm_eval import evaluator

    # ---- Build model arguments ----
    if backend == 'vllm':
        model_args_parts = [
            f'pretrained={args.model}',
            f'dtype={args.precision}',
            f'tensor_parallel_size={args.tp}',
            f'gpu_memory_utilization={args.gpu_memory_utilization}',
        ]
        if args.max_model_len is not None:
            model_args_parts.append(f'max_model_len={args.max_model_len}')

        model_args = ','.join(model_args_parts)
        model_type = 'vllm'
        print(f'Using vllm backend: {model_args}')

    elif backend == 'hf':
        model_args_parts = [
            f'pretrained={args.model}',
            f'dtype={args.precision}',
            'trust_remote_code=True',
        ]

        model_args = ','.join(model_args_parts)
        model_type = 'hf'
        print(f'Using hf backend: {model_args}')

    else:
        print(f'Unknown backend: {backend}')
        exit(1)

    # ---- Parse tasks ----
    task_list = args.tasks.split(',')
    print(f'Tasks: {task_list}')
    print(f'Num fewshot: {args.num_fewshot}')
    print(f'Batch size: {args.bsz}')
    print()

    # ---- Run evaluation ----
    batch_size = args.bsz
    if batch_size != 'auto':
        try:
            batch_size = int(batch_size)
        except ValueError:
            pass  # keep as string (e.g. 'auto:4')

    results = evaluator.simple_evaluate(
        model=model_type,
        model_args=model_args,
        tasks=task_list,
        num_fewshot=args.num_fewshot,
        batch_size=batch_size,
        limit=args.limit,
        bootstrap_iters=10000,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
    )

    # ---- Print results ----
    print()
    print('=' * 60)
    print('Results:')
    print('=' * 60)
    for task_name, task_results in results['results'].items():
        print(f'\n  {task_name}:')
        for metric, value in sorted(task_results.items()):
            if isinstance(value, float):
                print(f'    {metric}: {value:.4f}')
            else:
                print(f'    {metric}: {value}')

    # ---- Save results ----
    if args.output_path:
        with open(args.output_path, 'w') as f:
            json.dump(results['results'], f, indent=2, ensure_ascii=False)
        print(f'\nResults saved to {args.output_path}')

if __name__ == '__main__':
    main()
