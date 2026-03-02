#!/usr/bin/env python3
"""
Evaluate HuggingFace models using lm_eval with manual model loading.
Bypasses lm_eval's broken dtype handling in _create_model().
"""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator

TASKS = [
    "lambada_openai", "arc_easy", "arc_challenge",
    "hellaswag", "winogrande", "piqa", "openbookqa", "boolq",
]

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2-7B"
    bsz = sys.argv[2] if len(sys.argv) > 2 else "auto"

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print("Creating HFLM wrapper")
    lm = HFLM(pretrained=model, tokenizer=tokenizer)

    print(f"Running evaluation on: {TASKS}")
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=TASKS,
        batch_size=int(bsz) if bsz != "auto" else bsz,
        num_fewshot=0,
    )

    # Print results table
    print("\n" + "=" * 70)
    print(f" Results: {model_name}")
    print("=" * 70)
    if "results" in results:
        for task, metrics in results["results"].items():
            print(f"\n  {task}:")
            for k, v in metrics.items():
                if k != "alias" and not k.endswith("_stderr"):
                    stderr_key = k + "_stderr"
                    stderr = metrics.get(stderr_key, "")
                    stderr_str = f" ± {stderr:.4f}" if isinstance(stderr, float) else ""
                    if isinstance(v, float):
                        print(f"    {k}: {v:.4f}{stderr_str}")
                    else:
                        print(f"    {k}: {v}{stderr_str}")
    print("=" * 70)

if __name__ == "__main__":
    main()
