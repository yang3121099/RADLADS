#!/usr/bin/env python3
"""
Chatbot 模型测评脚本
使用 lm_eval harness 运行 chatbot 相关的 benchmark:
  - IFEval (instruction following)
  - TruthfulQA (mc2)
  - MMLU (知识广度)
  - ARC-Challenge (推理)
  - HellaSwag (常识推理)
  - Winogrande (共指消解)
  - GSM8K (数学)

用法:
    python eval_chatbot.py eval --path out/.../rwkv-step150-20M.pth --gpu 0
    python eval_chatbot.py eval_all --dir out/L28-D3584-qwerky7_qwen2-6_chatbot_sft --gpu 0
    python eval_chatbot.py eval_baseline --model Qwen/Qwen2.5-7B-Instruct --gpu 0
    python eval_chatbot.py summary
"""
import argparse
import json
import os
import sys
import glob
import re
from datetime import datetime
from pathlib import Path

EVAL_LOG = "eval_chatbot_results.json"
SUMMARY_MD = "eval_chatbot_summary.md"

# Chatbot benchmark tasks
CHATBOT_TASKS = "arc_challenge,hellaswag,winogrande,truthfulqa_mc2,mmlu,gsm8k"

# RADLADS model eval args
COMMON_ARGS = [
    "-c", "configs/qwen7b.yaml",
    "-c", "configs/qwerky7.yaml",
    "--model.attention_type", "rwkv7_fla_chunk",
    "--model.ctx_len", "4096",
    "--precision", "bf16",
]

# Short column names for display
COL_SHORT = {
    "arc_challenge": "arc_c",
    "hellaswag": "hella",
    "winogrande": "wino",
    "truthfulqa_mc2": "tqa_mc2",
    "mmlu": "mmlu",
    "gsm8k": "gsm8k",
    "ifeval": "ifeval",
}


def load_log():
    if os.path.exists(EVAL_LOG):
        with open(EVAL_LOG, "r") as f:
            return json.load(f)
    return {}


def save_log(log):
    with open(EVAL_LOG, "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def get_ckpt_key(path):
    return os.path.abspath(path)


def is_eval_complete(log, ckpt_key):
    if ckpt_key not in log:
        return False
    return log[ckpt_key].get("done", False)


def extract_step_info(path):
    basename = os.path.basename(path)
    m = re.match(r"rwkv-step(\d+)-(\d+)M\.pth", basename)
    if m:
        return int(m.group(1)), f"{m.group(2)}M"
    if "final" in basename:
        return 999999, "final"
    m = re.match(r"rwkv-(\d+)\.pth", basename)
    if m:
        return int(m.group(1)), f"epoch{m.group(1)}"
    return 0, basename.replace(".pth", "")


def run_radlads_eval(path, tasks, bsz=1, env=None):
    """Run lm_eval on a RADLADS checkpoint."""
    import subprocess
    cmd = [
        sys.executable, "run_lm_eval.py",
        *COMMON_ARGS,
        "--path", path,
        "--tasks", tasks,
        "--bsz", str(bsz),
    ]
    print(f"\n[EVAL] Running chatbot benchmarks: {path}")
    print(f"[EVAL] Tasks: {tasks}")
    print(f"[EVAL] Command: {' '.join(cmd)}")

    if env is None:
        env = {**os.environ}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print(result.stdout)
    if result.stderr:
        for line in result.stderr.split('\n'):
            if 'error' in line.lower() or 'traceback' in line.lower():
                print(f"[STDERR] {line}")

    return _parse_lm_eval_results(result.stdout)


def run_hf_eval(model_name, tasks, bsz="auto", env=None):
    """Run lm_eval on a HuggingFace model."""
    import subprocess
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_name},dtype=bfloat16,trust_remote_code=True",
        "--tasks", tasks,
        "--batch_size", str(bsz),
        "--num_fewshot", "0",
    ]
    print(f"\n[EVAL] Running chatbot benchmarks on HF model: {model_name}")
    print(f"[EVAL] Tasks: {tasks}")
    print(f"[EVAL] Command: {' '.join(cmd)}")

    if env is None:
        env = {**os.environ}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print(result.stdout)

    return _parse_lm_eval_results(result.stdout)


def _parse_lm_eval_results(stdout):
    """Parse lm_eval results from stdout (dict format or table format)."""
    results = {}

    # Try parsing dict/OrderedDict format
    for line in stdout.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('{') or line.startswith("OrderedDict"):
            try:
                text = line.replace("OrderedDict(", "").rstrip(")")
                parsed = eval(text)
                for task_name, task_results in parsed.items():
                    if isinstance(task_results, dict):
                        # prefer acc_norm, then acc, then exact_match
                        for k in ['acc_norm,none', 'acc,none', 'exact_match,none',
                                   'prompt_level_strict_acc,none', 'inst_level_strict_acc,none']:
                            if k in task_results:
                                results[task_name] = round(task_results[k] * 100, 2)
                                break
            except Exception:
                pass

    # Also try table format: |task_name|N|metric|↑|value|±|stderr|
    if not results:
        for line in stdout.split('\n'):
            line = line.strip()
            m = re.match(r"\|\s*(\w+)\s*\|.*\|\s*(acc(?:_norm)?|exact_match)\s*\|.*\|\s*([\d.]+)\s*\|", line)
            if m:
                task = m.group(1)
                val = round(float(m.group(3)) * 100, 2)
                results[task] = val

    if not results:
        print("[WARN] Could not parse eval results from output")
    return results


def _save_result_locked(ckpt_key, path, results):
    """Thread-safe save."""
    import fcntl
    lock_path = EVAL_LOG + ".lock"
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        log = load_log()
        step, label = extract_step_info(path)
        if ckpt_key not in log:
            log[ckpt_key] = {
                "path": path,
                "basename": os.path.basename(path),
                "step": step,
                "label": label,
                "done": False,
                "results": {},
            }
        log[ckpt_key]["results"].update(results)
        log[ckpt_key]["done"] = True
        log[ckpt_key]["eval_time"] = datetime.now().isoformat()
        save_log(log)
        fcntl.flock(lock_f, fcntl.LOCK_UN)


def eval_checkpoint(path, bsz=1, force=False, gpu=None):
    """Evaluate a single RADLADS checkpoint."""
    log = load_log()
    ckpt_key = get_ckpt_key(path)

    if not force and is_eval_complete(log, ckpt_key):
        print(f"[SKIP] Already evaluated: {path}")
        return

    env = {**os.environ}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    try:
        results = run_radlads_eval(path, CHATBOT_TASKS, bsz, env=env)
        if results:
            _save_result_locked(ckpt_key, path, results)
            print(f"[OK] Chatbot benchmarks saved for {os.path.basename(path)}")
    except Exception as e:
        print(f"[ERROR] Eval failed: {e}")


def eval_all(directory, bsz=1, force=False, gpu=None):
    """Evaluate all checkpoints in a directory."""
    ckpts = sorted(glob.glob(os.path.join(directory, "rwkv-*.pth")))
    if not ckpts:
        print(f"[WARN] No checkpoints found in {directory}")
        return

    log = load_log()
    done = sum(1 for c in ckpts if is_eval_complete(log, get_ckpt_key(c)))
    total = len(ckpts)
    print(f"[INFO] Found {total} checkpoints, {done} already evaluated, {total - done} remaining")

    for i, ckpt in enumerate(ckpts):
        key = get_ckpt_key(ckpt)
        if not force and is_eval_complete(log, key):
            print(f"[SKIP] ({i+1}/{total}) {os.path.basename(ckpt)}")
            continue
        print(f"\n{'='*60}")
        print(f"[EVAL] ({i+1}/{total}) {os.path.basename(ckpt)}")
        print(f"{'='*60}")
        eval_checkpoint(ckpt, bsz, force, gpu)

    generate_summary()


def eval_baseline(model_name, bsz="auto", force=False, gpu=None):
    """Evaluate a HuggingFace baseline model."""
    log = load_log()
    ckpt_key = f"baseline:{model_name}"

    if not force and is_eval_complete(log, ckpt_key):
        print(f"[SKIP] Already evaluated baseline: {model_name}")
        return

    env = {**os.environ}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    try:
        results = run_hf_eval(model_name, CHATBOT_TASKS, bsz, env=env)
        if results:
            _save_result_locked(ckpt_key, model_name, results)
            print(f"[OK] Chatbot benchmarks saved for baseline {model_name}")
    except Exception as e:
        print(f"[ERROR] Baseline eval failed: {e}")


def generate_summary():
    """Generate markdown summary table."""
    log = load_log()
    if not log:
        print("[WARN] No evaluation results found")
        return

    entries = sorted(log.values(), key=lambda x: x.get("step", 0))

    # Collect all task names
    all_tasks = set()
    for entry in entries:
        all_tasks.update(entry.get("results", {}).keys())
    all_tasks = sorted(all_tasks)

    lines = []
    lines.append("# Chatbot Evaluation Results Summary\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Build table
    header = "| Checkpoint | Tokens |"
    separator = "|---|---|"
    for col in all_tasks:
        short = COL_SHORT.get(col, col)
        header += f" {short} |"
        separator += "---|"
    header += " avg |"
    separator += "---|"

    lines.append(header)
    lines.append(separator)

    for entry in entries:
        basename = entry.get("basename", "?")
        label = entry.get("label", "?")
        row = f"| {basename} | {label} |"

        values = []
        for col in all_tasks:
            val = entry.get("results", {}).get(col, None)
            if val is not None:
                row += f" {val:.1f} |"
                values.append(val)
            else:
                row += " - |"

        if values:
            avg = sum(values) / len(values)
            row += f" {avg:.1f} |"
        else:
            row += " - |"

        lines.append(row)

    md_content = "\n".join(lines) + "\n"

    with open(SUMMARY_MD, "w") as f:
        f.write(md_content)

    print(f"\n{'='*60}")
    print(md_content)
    print(f"Summary saved to {SUMMARY_MD}")


def main():
    parser = argparse.ArgumentParser(description="RADLADS Chatbot Eval Manager")
    sub = parser.add_subparsers(dest="command")

    p_eval = sub.add_parser("eval", help="Evaluate a single checkpoint")
    p_eval.add_argument("--path", required=True)
    p_eval.add_argument("--bsz", type=int, default=1)
    p_eval.add_argument("--force", action="store_true")
    p_eval.add_argument("--gpu", type=int, default=None)

    p_all = sub.add_parser("eval_all", help="Evaluate all checkpoints in directory")
    p_all.add_argument("--dir", required=True)
    p_all.add_argument("--bsz", type=int, default=1)
    p_all.add_argument("--force", action="store_true")
    p_all.add_argument("--gpu", type=int, default=None)

    p_base = sub.add_parser("eval_baseline", help="Evaluate HuggingFace baseline model")
    p_base.add_argument("--model", required=True)
    p_base.add_argument("--bsz", default="auto")
    p_base.add_argument("--force", action="store_true")
    p_base.add_argument("--gpu", type=int, default=None)

    p_sum = sub.add_parser("summary", help="Generate markdown summary")
    p_sum.add_argument("--log", default=None)

    args = parser.parse_args()

    if args.command == "eval":
        eval_checkpoint(args.path, args.bsz, args.force, gpu=args.gpu)
        generate_summary()
    elif args.command == "eval_all":
        eval_all(args.dir, args.bsz, args.force, gpu=args.gpu)
    elif args.command == "eval_baseline":
        eval_baseline(args.model, args.bsz, args.force, gpu=args.gpu)
        generate_summary()
    elif args.command == "summary":
        if args.log:
            global EVAL_LOG
            EVAL_LOG = args.log
        generate_summary()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
