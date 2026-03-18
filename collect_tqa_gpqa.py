#!/usr/bin/env python3
"""
从 eval log 文件中解析 truthfulqa_mc2 + gpqa_diamond_zeroshot 结果，生成 markdown 汇总表格。

用法:
    # 从 eval_logs/tqa_gpqa/ 目录下的 log 文件解析
    python collect_tqa_gpqa.py

    # 指定 log 目录
    python collect_tqa_gpqa.py --logdir eval_logs/tqa_gpqa

    # 指定 log 目录 + 也扫描 eval_logs/chatbot
    python collect_tqa_gpqa.py --logdir eval_logs/tqa_gpqa eval_logs/chatbot
"""
import os
import re
import sys
import json
import glob
import argparse


# 6 models to exclude
EXCLUDE_BASENAMES = {
    "L28-D3584-qwerky7_qwen2-6_chatbot_slimorca_rwkv-step1500-197M",
    "L28-D3584-qwerky7_qwen2-6_chatbot_slimorca_rwkv-step150-20M",
    "L28-D3584-qwerky7_qwen2-6_chatbot_openhermes_rwkv-step900-118M",
    "L28-D3584-qwerky7_qwen2-6_chatbot_ultrachat_rwkv-step1500-197M",
    "L28-D3584-qwerky7_qwen2-6_chatbot_ultrachat_rwkv-step150-20M",
    "L28-D3584-qwerky7_qwen2-4_BOA_rwkv-1",
}


def parse_results_from_text(text):
    """Parse lm_eval results dict from log text, handling np.float64 etc."""
    results = {}

    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Match lines that look like a Python dict with task results
        if line.startswith('{') or line.startswith("OrderedDict"):
            try:
                cleaned = line.replace("OrderedDict(", "").rstrip(")")
                # Strip numpy wrappers: np.float64(0.123) -> 0.123
                cleaned = re.sub(r'np\.\w+\(([^)]+)\)', r'\1', cleaned)
                parsed = eval(cleaned)
                for task_name, task_results in parsed.items():
                    if isinstance(task_results, dict):
                        # Priority order for metric selection
                        for k in ['acc_norm,none', 'acc,none', 'mc2,none',
                                   'exact_match,none', 'prompt_level_strict_acc,none']:
                            if k in task_results:
                                results[task_name] = round(float(task_results[k]) * 100, 2)
                                break
            except Exception as e:
                # Try regex fallback for partially broken lines
                pass

    # Regex fallback: extract individual metric values
    if not results:
        # Pattern: 'acc,none': 0.328 or 'acc_norm,none': np.float64(0.328)
        for task in ['truthfulqa_mc2', 'gpqa_diamond_zeroshot']:
            # Look for task dict block
            task_pattern = rf"'{task}':\s*\{{([^}}]+)\}}"
            m = re.search(task_pattern, text)
            if m:
                block = m.group(1)
                # Try acc_norm first, then acc, then mc2
                for metric in ['acc_norm,none', 'acc,none', 'mc2,none']:
                    val_pattern = rf"'{metric}':\s*(?:np\.float\d+\()?([\d.]+)\)?"
                    vm = re.search(val_pattern, block)
                    if vm:
                        results[task] = round(float(vm.group(1)) * 100, 2)
                        break

    return results


def extract_model_info(logfile):
    """Extract model directory and checkpoint name from log filename."""
    basename = os.path.basename(logfile).replace('.log', '')

    # Try to find the model path from log content
    try:
        with open(logfile, 'r', errors='replace') as f:
            head = f.read(2048)
        m = re.search(r'--path\s+(\S+)', head)
        if m:
            path = m.group(1)
            dirname = os.path.basename(os.path.dirname(path))
            ckpt_name = os.path.basename(path).replace('.pth', '')
            return dirname, ckpt_name, path
    except Exception:
        pass

    # Fallback: parse from filename (format: dirname_ckptname.log)
    # e.g. L28-D3584-qwerky7_qwen2-5_continue_rwkv-step150-20M.log
    # This is tricky because dirname itself contains underscores
    # Try to split on _rwkv-
    parts = basename.split('_rwkv-')
    if len(parts) == 2:
        dirname = parts[0]
        ckpt_name = 'rwkv-' + parts[1]
        return dirname, ckpt_name, ""

    return basename, "", ""


def extract_step_label(ckpt_name):
    """Extract step/token label from checkpoint name."""
    m = re.match(r"rwkv-step(\d+)-(\d+)M", ckpt_name)
    if m:
        return int(m.group(1)), f"{m.group(2)}M"
    if "final" in ckpt_name:
        return 999999, "final"
    m = re.match(r"rwkv-(\d+)", ckpt_name)
    if m:
        return int(m.group(1)), f"epoch{m.group(1)}"
    return 0, ckpt_name


def main():
    parser = argparse.ArgumentParser(description="Collect TQA+GPQA results from log files")
    parser.add_argument('--logdir', nargs='+', default=['eval_logs/tqa_gpqa', 'eval_logs/chatbot'],
                        help='Log directories to scan (default: eval_logs/tqa_gpqa eval_logs/chatbot)')
    parser.add_argument('--out', default='eval_tqa_gpqa_summary.md',
                        help='Output markdown file (default: eval_tqa_gpqa_summary.md)')
    args = parser.parse_args()

    # Collect all log files
    log_files = []
    for d in args.logdir:
        log_files.extend(glob.glob(os.path.join(d, '*.log')))
    log_files = sorted(set(log_files))

    if not log_files:
        print(f"[ERROR] No log files found in: {args.logdir}")
        sys.exit(1)

    print(f"Found {len(log_files)} log files")

    # Parse each log file
    entries = {}  # key -> {dirname, ckpt_name, step, label, truthfulqa_mc2, gpqa_diamond_zeroshot}

    for logfile in log_files:
        dirname, ckpt_name, path = extract_model_info(logfile)

        # Check exclusion
        log_basename = os.path.basename(logfile).replace('.log', '')
        if log_basename in EXCLUDE_BASENAMES or f"{dirname}_{ckpt_name}" in EXCLUDE_BASENAMES:
            continue

        try:
            with open(logfile, 'r', errors='replace') as f:
                text = f.read()
        except Exception as e:
            print(f"  [SKIP] Cannot read {logfile}: {e}")
            continue

        results = parse_results_from_text(text)
        if not results:
            continue

        tqa = results.get('truthfulqa_mc2')
        gpqa = results.get('gpqa_diamond_zeroshot')
        if tqa is None and gpqa is None:
            continue

        key = f"{dirname}/{ckpt_name}"
        step, label = extract_step_label(ckpt_name)

        # Merge with existing entry (in case results come from multiple log files)
        if key in entries:
            if tqa is not None:
                entries[key]['truthfulqa_mc2'] = tqa
            if gpqa is not None:
                entries[key]['gpqa_diamond_zeroshot'] = gpqa
        else:
            entries[key] = {
                'dirname': dirname,
                'ckpt_name': ckpt_name,
                'step': step,
                'label': label,
                'truthfulqa_mc2': tqa,
                'gpqa_diamond_zeroshot': gpqa,
            }
            print(f"  [OK] {key}: tqa={tqa}, gpqa={gpqa}")

    if not entries:
        print("[WARN] No results found in any log files")
        sys.exit(1)

    # Sort by dirname then step
    sorted_entries = sorted(entries.values(), key=lambda x: (x['dirname'], x['step']))

    # Generate markdown table
    lines = []
    lines.append("# TruthfulQA_MC2 + GPQA Diamond Evaluation Results\n")
    lines.append(f"Total models: {len(sorted_entries)}\n")
    lines.append("| Model | Tokens | truthfulqa_mc2 | gpqa_diamond | avg |")
    lines.append("|---|---|---|---|---|")

    for e in sorted_entries:
        short_dir = e['dirname']
        for prefix in ["L28-D3584-qwerky7_qwen2-", "L28-D3584-"]:
            if short_dir.startswith(prefix):
                short_dir = short_dir[len(prefix):]
                break
        display = f"{short_dir}/{e['ckpt_name']}"

        tqa = e.get('truthfulqa_mc2')
        gpqa = e.get('gpqa_diamond_zeroshot')

        tqa_str = f"{tqa:.1f}" if tqa is not None else "-"
        gpqa_str = f"{gpqa:.1f}" if gpqa is not None else "-"

        vals = [v for v in [tqa, gpqa] if v is not None]
        avg = sum(vals) / len(vals) if vals else 0
        avg_str = f"**{avg:.1f}**" if vals else "-"

        lines.append(f"| {display} | {e['label']} | {tqa_str} | {gpqa_str} | {avg_str} |")

    md = "\n".join(lines) + "\n"

    # Write file
    with open(args.out, 'w') as f:
        f.write(md)

    # Print to stdout
    print()
    print(md)
    print(f"Summary saved to {args.out}")


if __name__ == '__main__':
    main()
