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

    # 测评 out/ 下所有目录的所有 checkpoint（双卡并行，每卡3进程）
    python eval_manager.py eval_all_dirs

    # 测评 HuggingFace baseline 模型
    python eval_manager.py eval_baseline --model Qwen/Qwen2.5-7B

    # 生成汇总 markdown 表格
    python eval_manager.py summary

    # 从指定 JSON 文件生成汇总
    python eval_manager.py summary --log eval_result-bks.json

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


def eval_all_parallel(directory, bsz=4, force=False, gpu0=0, gpu1=1, workers_per_gpu=2):
    """Evaluate all checkpoints using 2 GPUs in parallel with multiple workers per GPU.

    Creates a task pool: each task is (checkpoint, eval_type).
    Tasks are distributed round-robin across GPUs, with up to `workers_per_gpu`
    concurrent processes per GPU.

    Example with workers_per_gpu=2:
      GPU0: [standard-ckpt1, standard-ckpt2] running simultaneously
      GPU1: [supergpqa-ckpt1, supergpqa-ckpt2] running simultaneously
    """
    import subprocess
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import threading

    ckpts = sorted(glob.glob(os.path.join(directory, "rwkv-*.pth")))
    if not ckpts:
        print(f"[WARN] No checkpoints found in {directory}")
        return

    log = load_log()

    # Build task list: (checkpoint_path, eval_type, gpu_id)
    tasks = []
    for ckpt in ckpts:
        key = get_ckpt_key(ckpt)
        if key not in log:
            log[key] = _init_log_entry(ckpt)
        entry = log.get(key, {})

        need_standard = force or not entry.get("standard_done", False)
        need_supergpqa = force or not entry.get("supergpqa_done", False)

        if need_standard:
            tasks.append((ckpt, "standard", gpu0))
        if need_supergpqa:
            tasks.append((ckpt, "supergpqa", gpu1))

    save_log(log)

    if not tasks:
        print(f"[INFO] All {len(ckpts)} checkpoints already evaluated, nothing to do")
        generate_summary()
        return

    total_tasks = len(tasks)
    total_ckpts = len(set(t[0] for t in tasks))
    total_workers = workers_per_gpu * 2
    print(f"[INFO] {total_tasks} tasks for {total_ckpts} checkpoints")
    print(f"[INFO] GPU{gpu0} -> standard, GPU{gpu1} -> SuperGPQA")
    print(f"[INFO] {workers_per_gpu} workers/GPU, {total_workers} total concurrent processes, bsz={bsz}")

    # Use semaphores to limit concurrency per GPU
    gpu_semaphores = {gpu0: threading.Semaphore(workers_per_gpu), gpu1: threading.Semaphore(workers_per_gpu)}

    def run_task(ckpt, eval_type, gpu_id):
        """Run a single eval task as a subprocess."""
        basename = os.path.basename(ckpt)
        sem = gpu_semaphores[gpu_id]
        sem.acquire()
        try:
            cmd = [
                sys.executable, "eval_manager.py", "eval",
                "--path", ckpt, "--bsz", str(bsz), "--gpu", str(gpu_id),
                "--only", eval_type,
            ]
            if force:
                cmd.append("--force")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
            print(f"  [GPU{gpu_id}] START {eval_type:10s} {basename}")
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
            print(f"  [GPU{gpu_id}]  DONE {eval_type:10s} {basename} [{status}]")
            if proc.returncode != 0 and proc.stderr:
                for line in proc.stderr.strip().split('\n')[-5:]:
                    print(f"         {line}")
            return (basename, eval_type, proc.returncode)
        finally:
            sem.release()

    # Submit all tasks to a thread pool (threads manage subprocesses + semaphores)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=total_workers) as pool:
        futures = {pool.submit(run_task, ckpt, etype, gpu): (ckpt, etype)
                   for ckpt, etype, gpu in tasks}

        done_count = 0
        failed = []
        for future in as_completed(futures):
            done_count += 1
            basename, eval_type, rc = future.result()
            if rc != 0:
                failed.append(f"{basename}/{eval_type}")
            if done_count % 4 == 0 or done_count == total_tasks:
                print(f"[PROGRESS] {done_count}/{total_tasks} tasks complete")

    if failed:
        print(f"\n[WARN] {len(failed)} tasks failed: {', '.join(failed)}")
    else:
        print(f"\n[OK] All {total_tasks} tasks completed successfully")

    # Auto-generate summary
    generate_summary()


def run_baseline_standard_eval(model_name, bsz="auto", gpu=None):
    """Run standard lm_eval benchmarks on a HuggingFace baseline model."""
    import subprocess
    env = {**os.environ}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    cmd = [
        sys.executable, "eval_qwen_hf.py", model_name, str(bsz),
    ]
    print(f"\n[EVAL] Running standard benchmarks on baseline: {model_name}")
    print(f"[EVAL] Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print(result.stdout)
    if result.stderr:
        for line in result.stderr.split('\n'):
            if 'error' in line.lower() or 'traceback' in line.lower():
                print(f"[STDERR] {line}")

    # Parse results - same format as eval_qwen_hf.py output
    results = {}
    current_task = None
    for line in result.stdout.split('\n'):
        line = line.strip()
        # Task header: "  lambada_openai:"
        m = re.match(r"(\w+):$", line)
        if m:
            current_task = m.group(1)
            continue
        # Metric line: "    acc,none: 0.7011"
        if current_task and line.startswith("acc"):
            m = re.match(r"(acc(?:_norm)?),none:\s+([\d.]+)", line)
            if m:
                # Use acc_norm if available, otherwise acc
                metric = m.group(1)
                val = round(float(m.group(2)) * 100, 2)
                # For tasks that use acc_norm (hellaswag, winogrande, arc_challenge), prefer it
                if metric == "acc_norm" or current_task not in results:
                    results[current_task] = val

    if not results:
        print("[WARN] Could not parse baseline standard eval results")

    return results


def run_baseline_supergpqa_eval(model_name, bsz="auto", gpu=None):
    """Run SuperGPQA eval on a HuggingFace baseline model using lm_eval."""
    import subprocess
    env = {**os.environ}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # Use lm_eval directly for HF models with SuperGPQA task
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_name},dtype=float16,trust_remote_code=True",
        "--tasks", "supergpqa",
        "--batch_size", str(bsz),
        "--num_fewshot", "0",
    ]
    print(f"\n[EVAL] Running SuperGPQA on baseline: {model_name}")
    print(f"[EVAL] Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print(result.stdout)

    # Parse SuperGPQA results from lm_eval output
    results = {}
    for line in result.stdout.split('\n'):
        line = line.strip()
        # lm_eval table format: |supergpqa|      0|acc     |↑  |0.1120|±  |0.0042|
        m = re.match(r"\|\s*supergpqa\s*\|.*\|\s*acc\s*\|.*\|\s*([\d.]+)\s*\|", line)
        if m:
            results["supergpqa_overall"] = round(float(m.group(1)) * 100, 2)

    if not results:
        print("[WARN] Could not parse baseline SuperGPQA results")

    return results


def eval_all_dirs_parallel(base_dir="out", bsz=4, force=False, gpu0=0, gpu1=1, workers_per_gpu=3, only=None):
    """Evaluate ALL checkpoint directories under base_dir using 2 GPUs in parallel.

    Each checkpoint = 1 worker. Models are round-robin distributed across GPUs,
    with up to workers_per_gpu concurrent models per GPU.

    Usage:
        python eval_manager.py eval_all_dirs
        python eval_manager.py eval_all_dirs --only standard --workers-per-gpu 3
    """
    import subprocess
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Collect all checkpoints across all dirs
    all_ckpts = []
    dirs = sorted(glob.glob(os.path.join(base_dir, "*")))
    for d in dirs:
        if not os.path.isdir(d):
            continue
        ckpts = sorted(glob.glob(os.path.join(d, "rwkv-*.pth")))
        if ckpts:
            all_ckpts.extend(ckpts)
            print(f"  {d}: {len(ckpts)} checkpoint(s)")

    if not all_ckpts:
        print(f"[ERROR] No checkpoints found under {base_dir}/")
        return

    log = load_log()

    # Build task list: (checkpoint_path, eval_type, gpu_id)
    # Round-robin GPU assignment per checkpoint
    tasks = []
    gpu_cycle = [gpu0, gpu1]
    for i, ckpt in enumerate(all_ckpts):
        key = get_ckpt_key(ckpt)
        if key not in log:
            log[key] = _init_log_entry(ckpt)
        entry = log.get(key, {})
        assigned_gpu = gpu_cycle[i % len(gpu_cycle)]

        if only in (None, "standard"):
            need_standard = force or not entry.get("standard_done", False)
            if need_standard:
                tasks.append((ckpt, "standard", assigned_gpu))

        if only in (None, "supergpqa"):
            need_supergpqa = force or not entry.get("supergpqa_done", False)
            if need_supergpqa:
                tasks.append((ckpt, "supergpqa", assigned_gpu))

    save_log(log)

    if not tasks:
        print(f"[INFO] All {len(all_ckpts)} checkpoints already evaluated, nothing to do")
        generate_summary()
        return

    total_tasks = len(tasks)
    total_ckpts = len(set(t[0] for t in tasks))
    total_workers = workers_per_gpu * 2
    eval_types = set(t[1] for t in tasks)
    # Log directory for per-model output files
    logdir = "eval_logs"
    os.makedirs(logdir, exist_ok=True)

    print(f"\n[INFO] {total_tasks} tasks for {total_ckpts} checkpoints")
    print(f"[INFO] Eval types: {', '.join(sorted(eval_types))}")
    print(f"[INFO] {workers_per_gpu} workers/GPU, {total_workers} total concurrent processes, bsz={bsz}")
    print(f"[INFO] Logs: {logdir}/<dir>_<checkpoint>_<eval_type>.log")
    print(f"[INFO] Monitor: tail -f {logdir}/*.log")

    # Use semaphores to limit concurrency per GPU
    gpu_semaphores = {gpu0: threading.Semaphore(workers_per_gpu), gpu1: threading.Semaphore(workers_per_gpu)}

    def run_task(ckpt, eval_type, gpu_id):
        """Run a single eval task as a subprocess."""
        basename = os.path.basename(ckpt).replace(".pth", "")
        dirbase = os.path.basename(os.path.dirname(ckpt))
        label = f"{dirbase}/{basename}"
        logfile = os.path.join(logdir, f"{dirbase}_{basename}_{eval_type}.log")
        sem = gpu_semaphores[gpu_id]
        sem.acquire()
        try:
            cmd = [
                sys.executable, "eval_manager.py", "eval",
                "--path", ckpt, "--bsz", str(bsz), "--gpu", str(gpu_id),
                "--only", eval_type,
            ]
            if force:
                cmd.append("--force")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
            print(f"  [GPU{gpu_id}] START {eval_type:10s} {label}")
            with open(logfile, "w") as lf:
                proc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, text=True)
            status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
            print(f"  [GPU{gpu_id}]  DONE {eval_type:10s} {label} [{status}]")
            if proc.returncode != 0:
                # Print last few lines from log on failure
                with open(logfile, "r") as lf:
                    lines = lf.readlines()
                    for line in lines[-5:]:
                        print(f"         {line.rstrip()}")
            return (label, eval_type, proc.returncode)
        finally:
            sem.release()

    # Submit all tasks to a thread pool
    with ThreadPoolExecutor(max_workers=total_workers) as pool:
        futures = {pool.submit(run_task, ckpt, etype, gpu): (ckpt, etype)
                   for ckpt, etype, gpu in tasks}

        done_count = 0
        failed = []
        for future in as_completed(futures):
            done_count += 1
            label, eval_type, rc = future.result()
            if rc != 0:
                failed.append(f"{label}/{eval_type}")
            if done_count % 4 == 0 or done_count == total_tasks:
                print(f"[PROGRESS] {done_count}/{total_tasks} tasks complete")

    if failed:
        print(f"\n[WARN] {len(failed)} tasks failed: {', '.join(failed)}")
    else:
        print(f"\n[OK] All {total_tasks} tasks completed successfully")

    generate_summary()


def eval_baseline(model_name, bsz="auto", force=False, gpu=None, only=None):
    """Evaluate a HuggingFace baseline model and save results to the same log."""
    log = load_log()
    ckpt_key = f"baseline:{model_name}"

    if not force and ckpt_key in log and is_eval_complete(log, ckpt_key):
        print(f"[SKIP] Already evaluated baseline: {model_name}")
        return

    if ckpt_key not in log:
        short_name = model_name.replace("/", "-")
        log[ckpt_key] = {
            "path": model_name,
            "basename": short_name,
            "step": -1,
            "label": "baseline",
            "standard_done": False,
            "supergpqa_done": False,
            "standard_results": {},
            "supergpqa_results": {},
        }
        save_log(log)

    entry = log[ckpt_key]

    # Standard benchmarks
    if only in (None, "standard") and (force or not entry.get("standard_done", False)):
        try:
            std_results = run_baseline_standard_eval(model_name, bsz, gpu)
            if std_results:
                _save_result(ckpt_key, model_name, "standard", std_results)
                print(f"[OK] Standard benchmarks saved for baseline {model_name}")
        except Exception as e:
            print(f"[ERROR] Baseline standard eval failed: {e}")

    # SuperGPQA
    if only in (None, "supergpqa") and (force or not entry.get("supergpqa_done", False)):
        try:
            sgpqa_results = run_baseline_supergpqa_eval(model_name, bsz, gpu)
            if sgpqa_results:
                _save_result(ckpt_key, model_name, "supergpqa", sgpqa_results)
                print(f"[OK] SuperGPQA saved for baseline {model_name}")
        except Exception as e:
            print(f"[ERROR] Baseline SuperGPQA eval failed: {e}")


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
    p_par.add_argument("--workers-per-gpu", type=int, default=2,
                       help="Number of concurrent eval processes per GPU (default: 2)")

    p_dirs = sub.add_parser("eval_all_dirs", help="Evaluate ALL checkpoint dirs under out/")
    p_dirs.add_argument("--base-dir", default="out", help="Base directory to scan (default: out)")
    p_dirs.add_argument("--bsz", type=int, default=4)
    p_dirs.add_argument("--force", action="store_true")
    p_dirs.add_argument("--gpu0", type=int, default=0)
    p_dirs.add_argument("--gpu1", type=int, default=1)
    p_dirs.add_argument("--workers-per-gpu", type=int, default=3,
                       help="Number of concurrent eval processes per GPU (default: 3)")
    p_dirs.add_argument("--only", choices=["standard", "supergpqa"], default=None,
                       help="Only run standard or supergpqa (default: both)")

    p_base = sub.add_parser("eval_baseline", help="Evaluate HuggingFace baseline model")
    p_base.add_argument("--model", required=True, help="HF model name, e.g. Qwen/Qwen2.5-7B")
    p_base.add_argument("--bsz", default="auto")
    p_base.add_argument("--force", action="store_true")
    p_base.add_argument("--gpu", type=int, default=None)
    p_base.add_argument("--only", choices=["standard", "supergpqa"], default=None)

    p_sum = sub.add_parser("summary", help="Generate markdown summary")
    p_sum.add_argument("--log", default=None, help="Path to eval log JSON (default: eval_results.json)")

    args = parser.parse_args()

    if args.command == "eval":
        eval_checkpoint(args.path, args.bsz, args.force, gpu=args.gpu, only=args.only)
        generate_summary()
    elif args.command == "eval_all":
        eval_all(args.dir, args.bsz, args.force)
    elif args.command == "eval_all_parallel":
        eval_all_parallel(args.dir, args.bsz, args.force, args.gpu0, args.gpu1, args.workers_per_gpu)
    elif args.command == "eval_all_dirs":
        eval_all_dirs_parallel(args.base_dir, args.bsz, args.force, args.gpu0, args.gpu1, args.workers_per_gpu, only=args.only)
    elif args.command == "eval_baseline":
        eval_baseline(args.model, args.bsz, args.force, gpu=args.gpu, only=args.only)
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
