#!/usr/bin/env python3
"""
从 eval_logs/ 下的 .log 文件解析测评结果，生成 markdown 表格。

用法:
    python parse_eval_logs.py                          # 解析 eval_logs/*.log
    python parse_eval_logs.py eval_logs/*BOA*.log      # 解析指定 log
    python parse_eval_logs.py --dir eval_logs          # 指定目录
"""
import re
import sys
import os
import glob
from datetime import datetime


# Standard benchmark tasks (display order)
TASK_ORDER = [
    "lambada_openai", "arc_easy", "arc_challenge",
    "hellaswag", "winogrande", "piqa", "openbookqa", "boolq",
]

SHORT_NAMES = {
    "lambada_openai": "lambada",
    "arc_easy": "arc_e",
    "arc_challenge": "arc_c",
    "hellaswag": "hella",
    "winogrande": "wino",
    "openbookqa": "obqa",
}


def parse_log(logfile):
    """Parse a single eval log file and extract task results.

    Supports two output formats:
    1. Python dict: {'task': {'acc,none': 0.xx, ...}, ...}
    2. lm_eval table: |task_name|N|metric|value|stderr|
    """
    results = {}
    model_path = None
    is_complete = False

    try:
        with open(logfile, "r", errors="replace") as f:
            text = f.read()
    except (FileNotFoundError, OSError):
        return None

    # Extract model path
    m = re.search(r"Loading model - (.+\.pth)", text)
    if m:
        model_path = m.group(1)

    # Check completion
    if "100%" in text and "batches" in text and "done" in text:
        is_complete = True

    # Method 1: Parse the final dict line (run_lm_eval.py prints results['results'])
    # Looks like: OrderedDict([('task', {'acc,none': 0.xx}), ...])
    # or: {'task': {'acc,none': 0.xx, ...}, ...}
    for line in reversed(text.split("\n")):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{") or line.startswith("OrderedDict"):
            try:
                clean = line.replace("OrderedDict(", "").rstrip(")")
                parsed = eval(clean)
                for task_name, task_data in parsed.items():
                    if isinstance(task_data, dict):
                        # Prefer acc_norm for tasks that use it
                        for key in ["acc_norm,none", "acc,none"]:
                            if key in task_data:
                                results[task_name] = round(task_data[key] * 100, 2)
                                break
                if results:
                    break
            except Exception:
                pass

    # Method 2: Parse lm_eval table format
    # |  task  |N| metric |value |stderr|
    if not results:
        for line in text.split("\n"):
            m = re.match(
                r"\|\s*(\w+)\s*\|.*\|\s*(acc(?:_norm)?)\s*\|.*\|\s*([\d.]+)\s*\|",
                line,
            )
            if m:
                task = m.group(1)
                val = round(float(m.group(3)) * 100, 2)
                # acc_norm takes priority over acc
                if task not in results or m.group(2) == "acc_norm":
                    results[task] = val

    if not results:
        return None

    return {
        "logfile": logfile,
        "model_path": model_path,
        "results": results,
        "is_complete": is_complete,
    }


def extract_label(logfile, model_path):
    """Extract a short label from the logfile name or model path."""
    base = os.path.basename(logfile).replace("_all.log", "").replace(".log", "")
    return base


def extract_sort_key(entry):
    """Sort by directory name, then by step/epoch number."""
    label = entry["label"]
    # Try to extract step number: rwkv-step300-39M -> 300
    m = re.search(r"step(\d+)", label)
    if m:
        return (label.rsplit("_rwkv", 1)[0], int(m.group(1)))
    # Try epoch: rwkv-1 -> 1
    m = re.search(r"rwkv-(\d+)", label)
    if m:
        return (label.rsplit("_rwkv", 1)[0], int(m.group(1)))
    return (label, 0)


def generate_summary(entries):
    """Generate markdown table from parsed entries."""
    # Collect all tasks that appear
    all_tasks = set()
    for e in entries:
        all_tasks.update(e["results"].keys())

    # Order tasks: known tasks first, then alphabetical for unknowns
    ordered = [t for t in TASK_ORDER if t in all_tasks]
    extra = sorted(all_tasks - set(TASK_ORDER))
    ordered += extra

    lines = []
    lines.append("# Evaluation Results Summary\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Header
    header = "| Model |"
    separator = "|---|"
    for t in ordered:
        short = SHORT_NAMES.get(t, t)
        header += f" {short} |"
        separator += "---|"
    header += " **avg** |"
    separator += "---|"

    lines.append(header)
    lines.append(separator)

    # Rows
    for entry in entries:
        label = entry["label"]
        row = f"| {label} |"
        values = []
        for t in ordered:
            val = entry["results"].get(t)
            if val is not None:
                row += f" {val:.1f} |"
                values.append(val)
            else:
                row += " - |"
        avg = sum(values) / len(values) if values else 0
        row += f" **{avg:.1f}** |"
        lines.append(row)

    return "\n".join(lines) + "\n"


def main():
    # Collect log files
    logfiles = []
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == "--dir":
                continue
            if os.path.isdir(arg):
                logfiles.extend(sorted(glob.glob(os.path.join(arg, "*.log"))))
            elif os.path.isfile(arg):
                logfiles.append(arg)
            else:
                # Try as glob pattern
                logfiles.extend(sorted(glob.glob(arg)))
    else:
        logfiles = sorted(glob.glob("eval_logs/*_all.log"))
        if not logfiles:
            logfiles = sorted(glob.glob("eval_logs/*.log"))

    if not logfiles:
        print("[ERROR] No log files found.")
        print("Usage:")
        print("  python parse_eval_logs.py                     # eval_logs/*.log")
        print("  python parse_eval_logs.py eval_logs/*BOA*.log  # specific logs")
        print("  python parse_eval_logs.py /path/to/dir         # all .log in dir")
        sys.exit(1)

    print(f"Scanning {len(logfiles)} log file(s)...\n")

    entries = []
    for lf in logfiles:
        parsed = parse_log(lf)
        if parsed is None:
            status = "no results"
            print(f"  [ ] {os.path.basename(lf):50s} {status}")
            continue
        label = extract_label(lf, parsed["model_path"])
        parsed["label"] = label
        n_tasks = len(parsed["results"])
        status = f"{n_tasks} tasks" + ("" if parsed["is_complete"] else " (incomplete)")
        print(f"  [x] {os.path.basename(lf):50s} {status}")
        entries.append(parsed)

    if not entries:
        print("\n[WARN] No parseable results found in any log file.")
        sys.exit(1)

    # Sort entries
    entries.sort(key=extract_sort_key)

    # Generate and print markdown
    md = generate_summary(entries)
    print(f"\n{'=' * 70}")
    print(md)

    # Save to file
    out_file = "eval_summary.md"
    with open(out_file, "w") as f:
        f.write(md)
    print(f"Saved to {out_file}")

    # Also print a compact terminal-friendly table
    print(f"\n{'=' * 70}")
    print("Compact view:\n")
    all_tasks = set()
    for e in entries:
        all_tasks.update(e["results"].keys())
    ordered = [t for t in TASK_ORDER if t in all_tasks]

    header = f"{'Model':45s}"
    for t in ordered:
        header += f" {SHORT_NAMES.get(t, t):>7s}"
    header += f" {'avg':>7s}"
    print(header)
    print("-" * len(header))

    for entry in entries:
        row = f"{entry['label']:45s}"
        vals = []
        for t in ordered:
            v = entry["results"].get(t)
            if v is not None:
                row += f" {v:7.1f}"
                vals.append(v)
            else:
                row += f" {'-':>7s}"
        avg = sum(vals) / len(vals) if vals else 0
        row += f" {avg:7.1f}"
        print(row)

    print()


if __name__ == "__main__":
    main()
