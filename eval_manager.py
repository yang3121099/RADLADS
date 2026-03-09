#!/usr/bin/env python3
"""
统一测评 + 结果管理脚本
- 测评结果保存到 JSON log (eval_results.json)
- 已完成测评的 checkpoint 自动跳过
- 汇总所有结果为 markdown 表格

用法:
    # 测评单个 checkpoint
    python eval_manager.py eval --path out/.../rwkv-step150-20M.pth

    # 测评目录下所有 checkpoint (跳过已完成的)
    python eval_manager.py eval_all --dir out/L28-D3584-qwerky7_qwen2-5_continue

    # 生成汇总 markdown 表格
    python eval_manager.py summary

    # 强制重新测评
    python eval_manager.py eval --path out/.../rwkv-step150-20M.pth --force
"""
import argparse
import json
import os
import sys
import glob
import re
from datetime import datetime
from pathlib import Path

EVAL_LOG = "eval_results.json"
SUMMARY_MD = "eval_summary.md"

# Standard benchmark tasks
STANDARD_TASKS = "lambada_openai,arc_easy,arc_challenge,hellaswag,winogrande,piqa,openbookqa"

# Common eval args
COMMON_ARGS = [
    "-c", "configs/qwen7b.yaml",
    "-c", "configs/qwerky7.yaml",
    "--model.attention_type", "rwkv7_fla_fused_recurrent",
    "--model.ctx_len", "4096",
    "--precision", "bf16",
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
    """Normalize checkpoint path to a stable key."""
    return os.path.abspath(path)


def is_eval_complete(log, ckpt_key):
    """Check if both standard and supergpqa evals are done."""
    if ckpt_key not in log:
        return False
    entry = log[ckpt_key]
    return entry.get("standard_done", False) and entry.get("supergpqa_done", False)


def extract_step_info(path):
    """Extract step number and token count from checkpoint filename."""
    basename = os.path.basename(path)
    # rwkv-step150-20M.pth
    m = re.match(r"rwkv-step(\d+)-(\d+)M\.pth", basename)
    if m:
        return int(m.group(1)), f"{m.group(2)}M"
    # rwkv-final.pth
    if "final" in basename:
        return 999999, "final"
    # rwkv-0.pth (epoch-based)
    m = re.match(r"rwkv-(\d+)\.pth", basename)
    if m:
        return int(m.group(1)), f"epoch{m.group(1)}"
    return 0, basename.replace(".pth", "")


def run_standard_eval(path, bsz=1, env=None):
    """Run standard lm_eval benchmarks and return results dict."""
    import subprocess
    cmd = [
        sys.executable, "run_lm_eval.py",
        *COMMON_ARGS,
        "--path", path,
        "--tasks", STANDARD_TASKS,
        "--bsz", str(bsz),
    ]
    print(f"\n[EVAL] Running standard benchmarks: {path}")
    print(f"[EVAL] Command: {' '.join(cmd)}")

    if env is None:
        env = {**os.environ}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print(result.stdout)
    if result.stderr:
        # Filter out common warnings, print important errors
        for line in result.stderr.split('\n'):
            if 'error' in line.lower() or 'traceback' in line.lower():
                print(f"[STDERR] {line}")

    # Parse results from stdout - lm_eval prints a dict at the end
    results = {}
    for line in result.stdout.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Try to parse the results dict that lm_eval prints
        if line.startswith('{') or line.startswith("OrderedDict"):
            try:
                # Handle OrderedDict format
                text = line.replace("OrderedDict(", "").rstrip(")")
                parsed = eval(text)  # safe here since we control the subprocess
                for task_name, task_results in parsed.items():
                    if isinstance(task_results, dict):
                        acc_key = None
                        for k in ['acc,none', 'acc_norm,none', 'acc']:
                            if k in task_results:
                                acc_key = k
                                break
                        if acc_key:
                            results[task_name] = round(task_results[acc_key] * 100, 2)
            except Exception:
                pass

    if not results:
        print("[WARN] Could not parse standard eval results from output")

    return results


def run_supergpqa_eval(path, bsz=1, env=None):
    """Run SuperGPQA eval and return results dict."""
    import subprocess
    cmd = [
        sys.executable, "eval_supergpqa.py",
        *COMMON_ARGS,
        "--path", path,
        "--bsz", str(bsz),
    ]
    print(f"\n[EVAL] Running SuperGPQA: {path}")
    print(f"[EVAL] Command: {' '.join(cmd)}")

    if env is None:
        env = {**os.environ}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print(result.stdout)
    if result.stderr:
        for line in result.stderr.split('\n'):
            if 'error' in line.lower() or 'traceback' in line.lower():
                print(f"[STDERR] {line}")

    # Parse SuperGPQA results
    results = {}
    for line in result.stdout.split('\n'):
        line = line.strip()
        # Overall Accuracy: 123/456 = 27.19%
        m = re.match(r"Overall Accuracy:\s+(\d+)/(\d+)\s+=\s+([\d.]+)%", line)
        if m:
            results["supergpqa_overall"] = float(m.group(3))
        # Difficulty lines:  easy      :  100/ 200 = 50.00%
        m = re.match(r"(\w+)\s*:\s+(\d+)/\s*(\d+)\s+=\s+([\d.]+)%", line)
        if m:
            key = m.group(1).strip().lower()
            if key in ["easy", "medium", "middle", "hard"]:
                results[f"supergpqa_{key}"] = float(m.group(4))

    if not results:
        print("[WARN] Could not parse SuperGPQA results from output")

    return results


def _init_log_entry(path):
    """Create a new log entry for a checkpoint."""
    step, label = extract_step_info(path)
    return {
        "path": path,
        "basename": os.path.basename(path),
        "step": step,
        "label": label,
        "standard_done": False,
        "supergpqa_done": False,
        "standard_results": {},
        "supergpqa_results": {},
    }


def _save_result(ckpt_key, path, eval_type, results):
    """Thread-safe save: reload log, merge result, write back."""
    import fcntl
    lock_path = EVAL_LOG + ".lock"
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        log = load_log()
        if ckpt_key not in log:
            log[ckpt_key] = _init_log_entry(path)
        entry = log[ckpt_key]
        if eval_type == "standard":
            entry["standard_results"] = results
            entry["standard_done"] = True
            entry["standard_eval_time"] = datetime.now().isoformat()
        elif eval_type == "supergpqa":
            entry["supergpqa_results"] = results
            entry["supergpqa_done"] = True
            entry["supergpqa_eval_time"] = datetime.now().isoformat()
        save_log(log)
        fcntl.flock(lock_f, fcntl.LOCK_UN)


def eval_checkpoint(path, bsz=1, force=False, gpu=None, only=None):
    """Evaluate a single checkpoint, saving results incrementally.

    Args:
        only: "standard" or "supergpqa" to run only one type, None for both
        gpu: CUDA device id to use (sets CUDA_VISIBLE_DEVICES)
    """
    log = load_log()
    ckpt_key = get_ckpt_key(path)

    if not force and is_eval_complete(log, ckpt_key):
        print(f"[SKIP] Already evaluated: {path}")
        return

    if ckpt_key not in log:
        log[ckpt_key] = _init_log_entry(path)
        save_log(log)

    entry = log[ckpt_key]

    env = {**os.environ}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # Standard benchmarks
    if only in (None, "standard") and (force or not entry.get("standard_done", False)):
        try:
            std_results = run_standard_eval(path, bsz, env=env)
            if std_results:
                _save_result(ckpt_key, path, "standard", std_results)
                print(f"[OK] Standard benchmarks saved for {os.path.basename(path)}")
        except Exception as e:
            print(f"[ERROR] Standard eval failed: {e}")

    # SuperGPQA
    if only in (None, "supergpqa") and (force or not entry.get("supergpqa_done", False)):
        try:
            sgpqa_results = run_supergpqa_eval(path, bsz, env=env)
            if sgpqa_results:
                _save_result(ckpt_key, path, "supergpqa", sgpqa_results)
                print(f"[OK] SuperGPQA saved for {os.path.basename(path)}")
        except Exception as e:
            print(f"[ERROR] SuperGPQA eval failed: {e}")


def eval_all(directory, bsz=1, force=False):
    """Evaluate all checkpoints in a directory, skipping completed ones."""
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
        eval_checkpoint(ckpt, bsz, force)

    # Auto-generate summary
    generate_summary()


def eval_all_parallel(directory, bsz=4, force=False, gpu0=0, gpu1=1):
    """Evaluate all checkpoints using 2 GPUs in parallel.

    GPU0 runs standard benchmarks, GPU1 runs SuperGPQA simultaneously.
    For each checkpoint, both evals run in parallel on separate GPUs.
    """
    import subprocess

    ckpts = sorted(glob.glob(os.path.join(directory, "rwkv-*.pth")))
    if not ckpts:
        print(f"[WARN] No checkpoints found in {directory}")
        return

    log = load_log()
    done = sum(1 for c in ckpts if is_eval_complete(log, get_ckpt_key(c)))
    total = len(ckpts)
    print(f"[INFO] Found {total} checkpoints, {done} already evaluated, {total - done} remaining")
    print(f"[INFO] Parallel mode: GPU{gpu0} -> standard, GPU{gpu1} -> SuperGPQA, bsz={bsz}")

    for i, ckpt in enumerate(ckpts):
        key = get_ckpt_key(ckpt)
        if not force and is_eval_complete(log, key):
            print(f"[SKIP] ({i+1}/{total}) {os.path.basename(ckpt)}")
            continue

        log = load_log()  # reload to check partial completions
        if key not in log:
            log[key] = _init_log_entry(ckpt)
            save_log(log)
        entry = log[key]

        print(f"\n{'='*60}")
        print(f"[EVAL] ({i+1}/{total}) {os.path.basename(ckpt)} [parallel]")
        print(f"{'='*60}")

        need_standard = force or not entry.get("standard_done", False)
        need_supergpqa = force or not entry.get("supergpqa_done", False)

        procs = []

        if need_standard:
            cmd_std = [
                sys.executable, "eval_manager.py", "eval",
                "--path", ckpt, "--bsz", str(bsz), "--gpu", str(gpu0),
                "--only", "standard",
            ]
            if force:
                cmd_std.append("--force")
            print(f"  [GPU{gpu0}] Standard benchmarks...")
            env0 = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu0)}
            p0 = subprocess.Popen(cmd_std, env=env0)
            procs.append(("standard", p0))

        if need_supergpqa:
            cmd_sgpqa = [
                sys.executable, "eval_manager.py", "eval",
                "--path", ckpt, "--bsz", str(bsz), "--gpu", str(gpu1),
                "--only", "supergpqa",
            ]
            if force:
                cmd_sgpqa.append("--force")
            print(f"  [GPU{gpu1}] SuperGPQA...")
            env1 = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu1)}
            p1 = subprocess.Popen(cmd_sgpqa, env=env1)
            procs.append(("supergpqa", p1))

        # Wait for both to finish
        for name, proc in procs:
            rc = proc.wait()
            if rc != 0:
                print(f"  [WARN] {name} exited with code {rc}")
            else:
                print(f"  [OK] {name} done")

    # Auto-generate summary
    generate_summary()


def generate_summary():
    """Generate markdown summary table from eval log."""
    log = load_log()
    if not log:
        print("[WARN] No evaluation results found")
        return

    # Sort entries by step number
    entries = sorted(log.values(), key=lambda x: x.get("step", 0))

    # Collect all task names
    std_tasks = set()
    sgpqa_keys = set()
    for entry in entries:
        std_tasks.update(entry.get("standard_results", {}).keys())
        sgpqa_keys.update(entry.get("supergpqa_results", {}).keys())
    std_tasks = sorted(std_tasks)
    sgpqa_keys = sorted(sgpqa_keys)

    all_cols = std_tasks + sgpqa_keys

    lines = []
    lines.append("# Evaluation Results Summary\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Build table
    header = "| Checkpoint | Tokens |"
    separator = "|---|---|"
    for col in all_cols:
        short = col.replace("lambada_openai", "lambada") \
                    .replace("arc_challenge", "arc_c") \
                    .replace("arc_easy", "arc_e") \
                    .replace("hellaswag", "hella") \
                    .replace("winogrande", "wino") \
                    .replace("openbookqa", "obqa") \
                    .replace("supergpqa_", "sgpqa_")
        header += f" {short} |"
        separator += "---|"

    # Add average column
    header += " avg |"
    separator += "---|"

    lines.append(header)
    lines.append(separator)

    for entry in entries:
        basename = entry.get("basename", "?")
        label = entry.get("label", "?")
        row = f"| {basename} | {label} |"

        values = []
        for col in all_cols:
            val = entry.get("standard_results", {}).get(col,
                  entry.get("supergpqa_results", {}).get(col, None))
            if val is not None:
                row += f" {val:.1f} |"
                values.append(val)
            else:
                row += " - |"

        # Average
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
    parser = argparse.ArgumentParser(description="RADLADS Eval Manager")
    sub = parser.add_subparsers(dest="command")

    p_eval = sub.add_parser("eval", help="Evaluate a single checkpoint")
    p_eval.add_argument("--path", required=True)
    p_eval.add_argument("--bsz", type=int, default=1)
    p_eval.add_argument("--force", action="store_true")
    p_eval.add_argument("--gpu", type=int, default=None)
    p_eval.add_argument("--only", choices=["standard", "supergpqa"], default=None)

    p_all = sub.add_parser("eval_all", help="Evaluate all checkpoints in directory")
    p_all.add_argument("--dir", required=True)
    p_all.add_argument("--bsz", type=int, default=1)
    p_all.add_argument("--force", action="store_true")

    p_par = sub.add_parser("eval_all_parallel", help="Evaluate all checkpoints using 2 GPUs in parallel")
    p_par.add_argument("--dir", required=True)
    p_par.add_argument("--bsz", type=int, default=4)
    p_par.add_argument("--force", action="store_true")
    p_par.add_argument("--gpu0", type=int, default=0)
    p_par.add_argument("--gpu1", type=int, default=1)

    p_sum = sub.add_parser("summary", help="Generate markdown summary")

    args = parser.parse_args()

    if args.command == "eval":
        eval_checkpoint(args.path, args.bsz, args.force, gpu=args.gpu, only=args.only)
        generate_summary()
    elif args.command == "eval_all":
        eval_all(args.dir, args.bsz, args.force)
    elif args.command == "eval_all_parallel":
        eval_all_parallel(args.dir, args.bsz, args.force, args.gpu0, args.gpu1)
    elif args.command == "summary":
        generate_summary()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
