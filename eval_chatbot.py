#!/usr/bin/env python3
"""
Chatbot 模型测评脚本 — 全量 benchmark 评测

=== 测评分组 ===

1. 基座保持指标 (base_retain) — 监控灾难性遗忘 [loglikelihood, 快]:
   - lambada_openai, hellaswag, winogrande, piqa
   这些指标应与基座模型保持接近，下降 >2% 说明训练过度

2. Chatbot 能力指标 (chatbot) — 衡量指令遵循和推理提升 [loglikelihood, 快]:
   - truthfulqa_mc2   : 真实性 (SFT 应提升)
   - arc_challenge     : 推理能力 (应保持或提升)
   - mmlu              : 知识广度 (应保持)
   - boolq             : 阅读理解 (应保持)

3. 进阶指标 (advanced) — 高难度知识推理 [loglikelihood, 快]:
   - mmlu_pro          : MMLU 加强版 (10选1, 更难)
   - gpqa_diamond_zeroshot : 博士级科学问答 (198题)

4. 生成式指标 (generative) — 需要 generate_until [慢]:
   - gsm8k             : 数学推理 (1319题, 生成+精确匹配)
   - ifeval            : 指令遵循 (541题, 规则评分, 最重要的 chatbot 指标)
   - bbh_zeroshot      : Big-Bench Hard (23子任务, 生成+精确匹配)

注: generative 组因 RWKV adapter 的 generate_until 为逐条生成，速度较慢。
    建议先跑 fast 组 (base_retain+chatbot+advanced)，再跑 generative 组。

用法:
    python eval_chatbot.py eval --path out/.../rwkv-step150-20M.pth
    python eval_chatbot.py eval --path out/.../rwkv-step150-20M.pth --group fast
    python eval_chatbot.py eval --path out/.../rwkv-step150-20M.pth --group generative
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
    # 基座保持 (loglikelihood, 快)
    "base_retain": "lambada_openai,hellaswag,winogrande,piqa",
    # Chatbot 核心 (loglikelihood, 快)
    "chatbot": "truthfulqa_mc2,arc_challenge,mmlu,boolq",
    # 进阶知识推理 (loglikelihood, 快)
    "advanced": "mmlu_pro,gpqa_diamond_zeroshot",
    # 新增指标 = chatbot + advanced (不含 base_retain)
    "new": "truthfulqa_mc2,mmlu,boolq,mmlu_pro,gpqa_diamond_zeroshot",
    # 生成式评测 (generate_until, 慢)
    "generative": "gsm8k,ifeval,bbh_zeroshot",
    # 快速全量 = base_retain + chatbot + advanced (全部 loglikelihood)
    "fast": "lambada_openai,hellaswag,winogrande,piqa,truthfulqa_mc2,arc_challenge,mmlu,boolq,mmlu_pro,gpqa_diamond_zeroshot",
    # 全量
    "all": "lambada_openai,hellaswag,winogrande,piqa,truthfulqa_mc2,arc_challenge,mmlu,boolq,mmlu_pro,gpqa_diamond_zeroshot,gsm8k,ifeval,bbh_zeroshot",
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
    "boolq": "boolq",
    "mmlu_pro": "mmlu_pro",
    "gpqa_diamond_zeroshot": "gpqa_d",
    "gsm8k": "gsm8k",
    "ifeval": "ifeval",
    "bbh_zeroshot": "bbh",
}

# Display order — grouped by category
TASK_ORDER = [
    # base retain (loglikelihood)
    "lambada_openai", "hellaswag", "winogrande", "piqa",
    # chatbot (loglikelihood)
    "truthfulqa_mc2", "arc_challenge", "mmlu", "boolq",
    # advanced (loglikelihood)
    "mmlu_pro", "gpqa_diamond_zeroshot",
    # generative (generate_until)
    "gsm8k", "ifeval", "bbh_zeroshot",
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
        return all(entry.get(f"{g}_done", False)
                   for g in ["base_retain", "chatbot", "advanced", "generative"])
    if group == "fast":
        return all(entry.get(f"{g}_done", False)
                   for g in ["base_retain", "chatbot", "advanced"])
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
        sys.executable, "-u", "run_lm_eval.py",
        *COMMON_ARGS,
        "--path", path,
        "--tasks", tasks,
        "--bsz", str(bsz),
    ]
    print(f"\n[EVAL] Running: {path}")
    print(f"[EVAL] Tasks: {tasks}")
    print(f"[EVAL] Command: {' '.join(cmd)}")
    sys.stdout.flush()

    if env is None:
        env = {**os.environ}
    env["PYTHONUNBUFFERED"] = "1"

    # Stream output in real-time (for log file progress) while capturing for parsing
    stdout_lines = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1, env=env)
    for line in proc.stdout:
        print(line, end='', flush=True)
        stdout_lines.append(line)
    stderr = proc.stderr.read()
    proc.wait()

    if stderr:
        for line in stderr.split('\n'):
            if 'error' in line.lower() or 'traceback' in line.lower():
                print(f"[STDERR] {line}", flush=True)

    return _parse_lm_eval_results(''.join(stdout_lines))


def run_hf_eval(model_name, tasks, bsz="auto", env=None):
    """Run lm_eval on a HuggingFace model."""
    import subprocess
    cmd = [
        sys.executable, "-u", "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_name},dtype=bfloat16,trust_remote_code=True",
        "--tasks", tasks,
        "--batch_size", str(bsz),
        "--num_fewshot", "0",
    ]
    print(f"\n[EVAL] Running HF model: {model_name}")
    print(f"[EVAL] Tasks: {tasks}")
    print(f"[EVAL] Command: {' '.join(cmd)}")
    sys.stdout.flush()

    if env is None:
        env = {**os.environ}
    env["PYTHONUNBUFFERED"] = "1"

    stdout_lines = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1, env=env)
    for line in proc.stdout:
        print(line, end='', flush=True)
        stdout_lines.append(line)
    stderr = proc.stderr.read()
    proc.wait()

    return _parse_lm_eval_results(''.join(stdout_lines))


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
                r"\|\s*([\w]+(?:_[\w]+)*)\s*\|.*\|\s*(acc(?:_norm)?|exact_match|mc2|prompt_level_strict_acc|inst_level_strict_acc)\s*\|.*\|\s*([\d.]+)\s*\|",
                line,
            )
            if m:
                task = m.group(1)
                metric = m.group(2)
                val = round(float(m.group(3)) * 100, 2)
                # Priority: acc_norm > mc2 > prompt_level_strict_acc > acc > exact_match
                priority = {"acc_norm": 5, "mc2": 4, "prompt_level_strict_acc": 3, "acc": 2, "exact_match": 1, "inst_level_strict_acc": 0}
                if task not in results or priority.get(metric, 0) > priority.get(results.get(f"_metric_{task}"), -1):
                    results[task] = val
                    results[f"_metric_{task}"] = priority.get(metric, 0)

    # Clean up internal metric tracking keys
    results = {k: v for k, v in results.items() if not k.startswith("_metric_")}

    # Aggregate BBH subtasks into bbh_zeroshot average if present
    bbh_keys = [k for k in results if k.startswith("bbh_") and k != "bbh_zeroshot"]
    if bbh_keys:
        results["bbh_zeroshot"] = round(sum(results[k] for k in bbh_keys) / len(bbh_keys), 2)

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
                "advanced_done": False,
                "generative_done": False,
                "results": {},
            }
        log[ckpt_key]["results"].update(results)
        log[ckpt_key][f"{group}_done"] = True
        log[ckpt_key][f"{group}_eval_time"] = datetime.now().isoformat()
        # Mark all done if all groups complete
        all_groups = ["base_retain", "chatbot", "advanced", "generative"]
        if all(log[ckpt_key].get(f"{g}_done", False) for g in all_groups):
            log[ckpt_key]["done"] = True
        save_log(log)
        fcntl.flock(lock_f, fcntl.LOCK_UN)


def eval_checkpoint(path, bsz=4, force=False, gpu=None, group="all", tasks_override=None):
    """Evaluate a single RADLADS checkpoint."""
    log = load_log()
    ckpt_key = get_ckpt_key(path)

    env = {**os.environ}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    if tasks_override:
        # Direct task specification — run as a custom group, save under "custom"
        print(f"[EVAL] Custom tasks: {tasks_override}")
        try:
            results = run_radlads_eval(path, tasks_override, bsz, env=env)
            if results:
                _save_result_locked(ckpt_key, path, "custom", results)
                print(f"[OK] custom benchmarks saved for {os.path.basename(path)}")
        except Exception as e:
            print(f"[ERROR] custom eval failed: {e}")
        return

    groups_to_run = []
    if group == "all":
        groups_to_run = ["base_retain", "chatbot", "advanced", "generative"]
    elif group == "fast":
        groups_to_run = ["base_retain", "chatbot", "advanced"]
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

    if group == "all":
        groups_to_run = ["base_retain", "chatbot", "advanced", "generative"]
    elif group == "fast":
        groups_to_run = ["base_retain", "chatbot", "advanced"]
    else:
        groups_to_run = [group]

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
    lines.append("- **Chatbot** (tqa_mc2, arc_c, mmlu, boolq): 指令遵循/推理, SFT 应提升或保持")
    lines.append("- **Advanced** (mmlu_pro, gpqa_d): 高难度知识推理")
    lines.append("- **Generative** (gsm8k, ifeval, bbh): 生成式评测 (数学/指令遵循/推理)\n")

    # Build table
    # Determine group boundaries
    base_retain_tasks = [t for t in TASK_ORDER[:4] if t in all_present]
    chatbot_tasks = [t for t in TASK_ORDER[4:8] if t in all_present]
    advanced_tasks = [t for t in TASK_ORDER[8:10] if t in all_present]
    generative_tasks = [t for t in TASK_ORDER[10:] if t in all_present]

    header = "| Model | Tokens |"
    separator = "|---|---|"
    for t in ordered_tasks:
        header += f" {COL_SHORT.get(t, t)} |"
        separator += "---|"
    header += " base_avg | chat_avg | adv_avg | gen_avg | **total** |"
    separator += "---|---|---|---|---|"

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
        adv_vals = []
        gen_vals = []
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
                elif t in chatbot_tasks:
                    chat_vals.append(val)
                elif t in advanced_tasks:
                    adv_vals.append(val)
                elif t in generative_tasks:
                    gen_vals.append(val)
            else:
                row += " - |"

        base_avg = sum(base_vals) / len(base_vals) if base_vals else 0
        chat_avg = sum(chat_vals) / len(chat_vals) if chat_vals else 0
        adv_avg = sum(adv_vals) / len(adv_vals) if adv_vals else 0
        gen_avg = sum(gen_vals) / len(gen_vals) if gen_vals else 0
        all_vals = base_vals + chat_vals + adv_vals + gen_vals
        total_avg = sum(all_vals) / len(all_vals) if all_vals else 0
        row += f" {base_avg:.1f} | {chat_avg:.1f} | {adv_avg:.1f} | {gen_avg:.1f} | **{total_avg:.1f}** |"

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
    p_eval.add_argument("--group", choices=list(TASK_GROUPS.keys()), default="all",
                        help="Which task group to evaluate (default: all)")
    p_eval.add_argument("--tasks", type=str, default=None,
                        help="Override: comma-separated task names (bypasses --group)")

    p_all = sub.add_parser("eval_all", help="Evaluate all checkpoints in directory")
    p_all.add_argument("--dir", required=True)
    p_all.add_argument("--bsz", type=int, default=4)
    p_all.add_argument("--force", action="store_true")
    p_all.add_argument("--gpu", type=int, default=None)
    p_all.add_argument("--group", choices=list(TASK_GROUPS.keys()), default="all")

    p_base = sub.add_parser("eval_baseline", help="Evaluate HuggingFace baseline model")
    p_base.add_argument("--model", required=True)
    p_base.add_argument("--bsz", default="auto")
    p_base.add_argument("--force", action="store_true")
    p_base.add_argument("--gpu", type=int, default=None)
    p_base.add_argument("--group", choices=list(TASK_GROUPS.keys()), default="all")

    p_sum = sub.add_parser("summary", help="Generate markdown summary")
    p_sum.add_argument("--log", default=None)

    args = parser.parse_args()

    if args.command == "eval":
        eval_checkpoint(args.path, args.bsz, args.force, gpu=args.gpu, group=args.group,
                        tasks_override=args.tasks)
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
