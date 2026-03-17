#!/usr/bin/env python3
"""
Chatbot 模型测评脚本 — 面向指令遵循和对话能力的 benchmark 组合

=== 测评分组 ===

1. 基座保持指标 (base_retain) — 监控灾难性遗忘:
   - lambada_openai, hellaswag, winogrande, piqa
   这些指标应与基座模型保持接近，下降 >2% 说明训练过度

2. Chatbot 能力指标 (chatbot) — 衡量指令遵循和推理提升:
   - truthfulqa_mc2   : 真实性 (SFT 应提升)
   - arc_challenge     : 推理能力 (应保持或提升)
   - mmlu              : 知识广度 (应保持)
   - gsm8k             : 数学推理 (应保持或提升)
   - boolq             : 阅读理解 (应保持)

注: IFEval 需要生成式评测，当前 RWKV adapter 的 generate_until 较慢，
    暂不纳入自动化流程。如需手动测试指令遵循，可单独运行。

用法:
    python eval_chatbot.py eval --path out/.../rwkv-step150-20M.pth
    python eval_chatbot.py eval --path out/.../rwkv-step150-20M.pth --group chatbot
    python eval_chatbot.py eval_all --dir out/L28-D3584-qwerky7_qwen2-6_chatbot_ultrachat
    python eval_chatbot.py eval_baseline --model Qwen/Qwen2.5-7B-Instruct
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

# Task groups
TASK_GROUPS = {
    "base_retain": "lambada_openai,hellaswag,winogrande,piqa",
    "chatbot": "truthfulqa_mc2,arc_challenge,mmlu,gsm8k,boolq",
    "all": "lambada_openai,hellaswag,winogrande,piqa,truthfulqa_mc2,arc_challenge,mmlu,gsm8k,boolq",
}

# RADLADS model eval args
COMMON_ARGS = [
    "-c", "configs/qwen7b.yaml",
    "-c", "configs/qwerky7.yaml",
    "--model.attention_type", "rwkv7_fla_fused_recurrent",
    "--model.ctx_len", "4096",
    "--precision", "bf16",
]

# Column display names
COL_SHORT = {
    "lambada_openai": "lambada",
    "hellaswag": "hella",
    "winogrande": "wino",
    "piqa": "piqa",
    "truthfulqa_mc2": "tqa_mc2",
    "arc_challenge": "arc_c",
    "mmlu": "mmlu",
    "gsm8k": "gsm8k",
    "boolq": "boolq",
}

# Display order
TASK_ORDER = [
    # base retain
    "lambada_openai", "hellaswag", "winogrande", "piqa",
    # chatbot
    "truthfulqa_mc2", "arc_challenge", "mmlu", "gsm8k", "boolq",
]


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


def is_eval_complete(log, ckpt_key, group="all"):
    if ckpt_key not in log:
        return False
    entry = log[ckpt_key]
    if group == "all":
        return entry.get("base_retain_done", False) and entry.get("chatbot_done", False)
    return entry.get(f"{group}_done", False)


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
    print(f"\n[EVAL] Running: {path}")
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
    print(f"\n[EVAL] Running HF model: {model_name}")
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
                        for k in ['acc_norm,none', 'acc,none', 'exact_match,none',
                                   'mc2,none', 'prompt_level_strict_acc,none',
                                   'inst_level_strict_acc,none']:
                            if k in task_results:
                                results[task_name] = round(task_results[k] * 100, 2)
                                break
            except Exception:
                pass

    # Also try table format: |task_name|N|metric|↑|value|±|stderr|
    if not results:
        for line in stdout.split('\n'):
            line = line.strip()
            m = re.match(
                r"\|\s*(\w+)\s*\|.*\|\s*(acc(?:_norm)?|exact_match|mc2)\s*\|.*\|\s*([\d.]+)\s*\|",
                line,
            )
            if m:
                task = m.group(1)
                metric = m.group(2)
                val = round(float(m.group(3)) * 100, 2)
                # acc_norm takes priority over acc
                if task not in results or metric in ("acc_norm", "mc2"):
                    results[task] = val

    if not results:
        print("[WARN] Could not parse eval results from output")
    return results


def _save_result_locked(ckpt_key, path, group, results):
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
                "dir": os.path.basename(os.path.dirname(path)),
                "step": step,
                "label": label,
                "base_retain_done": False,
                "chatbot_done": False,
                "results": {},
            }
        log[ckpt_key]["results"].update(results)
        log[ckpt_key][f"{group}_done"] = True
        log[ckpt_key][f"{group}_eval_time"] = datetime.now().isoformat()
        # Mark all done if both groups complete
        if log[ckpt_key].get("base_retain_done") and log[ckpt_key].get("chatbot_done"):
            log[ckpt_key]["done"] = True
        save_log(log)
        fcntl.flock(lock_f, fcntl.LOCK_UN)


def eval_checkpoint(path, bsz=4, force=False, gpu=None, group="all"):
    """Evaluate a single RADLADS checkpoint."""
    log = load_log()
    ckpt_key = get_ckpt_key(path)

    env = {**os.environ}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    groups_to_run = []
    if group == "all":
        groups_to_run = ["base_retain", "chatbot"]
    else:
        groups_to_run = [group]

    for g in groups_to_run:
        if not force and is_eval_complete(log, ckpt_key, g):
            print(f"[SKIP] {g} already evaluated: {path}")
            continue
        tasks = TASK_GROUPS[g]
        try:
            results = run_radlads_eval(path, tasks, bsz, env=env)
            if results:
                _save_result_locked(ckpt_key, path, g, results)
                print(f"[OK] {g} benchmarks saved for {os.path.basename(path)}")
        except Exception as e:
            print(f"[ERROR] {g} eval failed: {e}")


def eval_all(directory, bsz=4, force=False, gpu=None, group="all"):
    """Evaluate all checkpoints in a directory."""
    ckpts = sorted(glob.glob(os.path.join(directory, "rwkv-*.pth")))
    if not ckpts:
        print(f"[WARN] No checkpoints found in {directory}")
        return

    log = load_log()
    done = sum(1 for c in ckpts if is_eval_complete(log, get_ckpt_key(c), group))
    total = len(ckpts)
    print(f"[INFO] Found {total} checkpoints, {done} already evaluated, {total - done} remaining")

    for i, ckpt in enumerate(ckpts):
        key = get_ckpt_key(ckpt)
        if not force and is_eval_complete(log, key, group):
            print(f"[SKIP] ({i+1}/{total}) {os.path.basename(ckpt)}")
            continue
        print(f"\n{'='*60}")
        print(f"[EVAL] ({i+1}/{total}) {os.path.basename(ckpt)}")
        print(f"{'='*60}")
        eval_checkpoint(ckpt, bsz, force, gpu, group)

    generate_summary()


def eval_baseline(model_name, bsz="auto", force=False, gpu=None, group="all"):
    """Evaluate a HuggingFace baseline model."""
    log = load_log()
    ckpt_key = f"baseline:{model_name}"

    env = {**os.environ}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    groups_to_run = ["base_retain", "chatbot"] if group == "all" else [group]

    for g in groups_to_run:
        if not force and ckpt_key in log and log[ckpt_key].get(f"{g}_done"):
            print(f"[SKIP] {g} already evaluated for baseline: {model_name}")
            continue
        tasks = TASK_GROUPS[g]
        try:
            results = run_hf_eval(model_name, tasks, bsz, env=env)
            if results:
                _save_result_locked(ckpt_key, model_name, g, results)
                print(f"[OK] {g} benchmarks saved for baseline {model_name}")
        except Exception as e:
            print(f"[ERROR] Baseline {g} eval failed: {e}")


def generate_summary():
    """Generate markdown summary with grouped columns and delta from base."""
    log = load_log()
    if not log:
        print("[WARN] No evaluation results found")
        return

    # Sort: baselines first (step=-1), then by dir name, then step
    entries = sorted(log.values(), key=lambda x: (
        0 if x.get("step", 0) < 0 else 1,
        x.get("dir", ""),
        x.get("step", 0),
    ))

    # Find base model entry for delta calculation
    base_entry = None
    for e in entries:
        if "BOA" in e.get("dir", "") or "BOA" in e.get("basename", ""):
            base_entry = e
            break

    # Collect all tasks present
    all_present = set()
    for entry in entries:
        all_present.update(entry.get("results", {}).keys())

    # Use ordered task list, only include tasks that exist
    ordered_tasks = [t for t in TASK_ORDER if t in all_present]

    lines = []
    lines.append("# Chatbot SFT Evaluation Results\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## Task Groups\n")
    lines.append("- **Base Retain** (lambada, hella, wino, piqa): 应与基座保持接近, 下降>2%说明过拟合")
    lines.append("- **Chatbot** (tqa_mc2, arc_c, mmlu, gsm8k, boolq): 指令遵循/推理, SFT 应提升或保持\n")

    # Build table
    # Determine group boundaries
    base_retain_tasks = [t for t in TASK_ORDER[:4] if t in all_present]
    chatbot_tasks = [t for t in TASK_ORDER[4:] if t in all_present]

    header = "| Model | Tokens |"
    separator = "|---|---|"
    for t in ordered_tasks:
        header += f" {COL_SHORT.get(t, t)} |"
        separator += "---|"
    header += " base_avg | chat_avg | **total** |"
    separator += "---|---|---|"

    lines.append(header)
    lines.append(separator)

    for entry in entries:
        basename = entry.get("basename", "?")
        dirname = entry.get("dir", "")
        label = entry.get("label", "?")

        # Shorten display name
        short_dir = dirname.replace("L28-D3584-qwerky7_qwen2-6_chatbot_", "").replace("L28-D3584-qwerky7_qwen2-4_", "")
        if short_dir:
            display = f"{short_dir}/{basename}"
        else:
            display = basename

        row = f"| {display} | {label} |"

        base_vals = []
        chat_vals = []
        for t in ordered_tasks:
            val = entry.get("results", {}).get(t)
            if val is not None:
                # Show delta from base if available
                delta_str = ""
                if base_entry and base_entry is not entry:
                    base_val = base_entry.get("results", {}).get(t)
                    if base_val is not None:
                        delta = val - base_val
                        if abs(delta) >= 0.5:
                            sign = "+" if delta > 0 else ""
                            delta_str = f" ({sign}{delta:.1f})"
                row += f" {val:.1f}{delta_str} |"
                if t in base_retain_tasks:
                    base_vals.append(val)
                if t in chatbot_tasks:
                    chat_vals.append(val)
            else:
                row += " - |"

        base_avg = sum(base_vals) / len(base_vals) if base_vals else 0
        chat_avg = sum(chat_vals) / len(chat_vals) if chat_vals else 0
        all_vals = base_vals + chat_vals
        total_avg = sum(all_vals) / len(all_vals) if all_vals else 0
        row += f" {base_avg:.1f} | {chat_avg:.1f} | **{total_avg:.1f}** |"

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
    p_eval.add_argument("--bsz", type=int, default=4)
    p_eval.add_argument("--force", action="store_true")
    p_eval.add_argument("--gpu", type=int, default=None)
    p_eval.add_argument("--group", choices=["all", "base_retain", "chatbot"], default="all",
                        help="Which task group to evaluate (default: all)")

    p_all = sub.add_parser("eval_all", help="Evaluate all checkpoints in directory")
    p_all.add_argument("--dir", required=True)
    p_all.add_argument("--bsz", type=int, default=4)
    p_all.add_argument("--force", action="store_true")
    p_all.add_argument("--gpu", type=int, default=None)
    p_all.add_argument("--group", choices=["all", "base_retain", "chatbot"], default="all")

    p_base = sub.add_parser("eval_baseline", help="Evaluate HuggingFace baseline model")
    p_base.add_argument("--model", required=True)
    p_base.add_argument("--bsz", default="auto")
    p_base.add_argument("--force", action="store_true")
    p_base.add_argument("--gpu", type=int, default=None)
    p_base.add_argument("--group", choices=["all", "base_retain", "chatbot"], default="all")

    p_sum = sub.add_parser("summary", help="Generate markdown summary")
    p_sum.add_argument("--log", default=None)

    args = parser.parse_args()

    if args.command == "eval":
        eval_checkpoint(args.path, args.bsz, args.force, gpu=args.gpu, group=args.group)
        generate_summary()
    elif args.command == "eval_all":
        eval_all(args.dir, args.bsz, args.force, gpu=args.gpu, group=args.group)
    elif args.command == "eval_baseline":
        eval_baseline(args.model, args.bsz, args.force, gpu=args.gpu, group=args.group)
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
