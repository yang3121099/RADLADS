"""
Reasoning 数据集预处理脚本
将 HuggingFace 上的 reasoning 数据集转换为 RADLADS 训练所需的 binidx 格式

用法:
    python prepare_reasoning_data.py --dataset open-r1/OpenR1-Math-220k --ctxlen 4096
    python prepare_reasoning_data.py --dataset open-thoughts/OpenThoughts-114k --ctxlen 4096
    python prepare_reasoning_data.py --dataset AI-MO/NuminaMath-CoT --ctxlen 4096
    python prepare_reasoning_data.py --dataset bespokelabs/Bespoke-Stratos-17k --ctxlen 4096
"""

import argparse
import json
import os
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

# ============================================================================
# binidx builder (from make_data_hf.py)
# ============================================================================

from src.binidx import MMapIndexedDataset

def index_file_path(prefix_path):
    return prefix_path + ".idx"

class MMapIndexedDatasetBuilder:
    def __init__(self, bin_filename, dtype=np.int32):
        self._data_file = open(bin_filename, "wb")
        self._dtype = dtype
        self._sizes = []
        self._doc_idx = [0]
        self._token_count = 0

    def token_count(self):
        return self._token_count

    def add_doc(self, tokens):
        arr = np.asarray(tokens, dtype=self._dtype)
        self._token_count += len(arr)
        self._data_file.write(arr.tobytes(order="C"))
        self._sizes.append(len(arr))

    def finalize(self, index_file):
        self._data_file.close()
        self._doc_idx = list(range(len(self._sizes)))
        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)

# ============================================================================
# Dataset formatters
# ============================================================================

def format_openr1_math(example):
    """open-r1/OpenR1-Math-220k: problem + first correct generation"""
    problem = example.get("problem", "")
    solution = example.get("solution", "")
    generations = example.get("generations", [])
    correctness = example.get("correctness_math_verify", [])

    # pick first verified-correct generation, else fallback to solution
    reasoning = ""
    if generations and correctness:
        for gen, correct in zip(generations, correctness):
            if correct:
                reasoning = gen
                break
    if not reasoning:
        reasoning = solution if solution else (generations[0] if generations else "")

    if not problem or not reasoning:
        return None
    return f"Problem:\n{problem}\n\nSolution:\n{reasoning}"


def format_openthoughts(example):
    """open-thoughts/OpenThoughts-114k (metadata config)"""
    problem = example.get("problem", "")
    reasoning = example.get("deepseek_reasoning", "")
    solution = example.get("deepseek_solution", "")

    if not problem:
        return None
    text = f"Problem:\n{problem}"
    if reasoning:
        text += f"\n\nReasoning:\n{reasoning}"
    if solution:
        text += f"\n\nSolution:\n{solution}"
    return text


def format_numina_cot(example):
    """AI-MO/NuminaMath-CoT"""
    problem = example.get("problem", "")
    solution = example.get("solution", "")
    if not problem or not solution:
        return None
    return f"Problem:\n{problem}\n\nSolution:\n{solution}"


def format_stratos(example):
    """bespokelabs/Bespoke-Stratos-17k"""
    conversations = example.get("conversations", [])
    if not conversations:
        return None
    parts = []
    for msg in conversations:
        role = msg.get("from", msg.get("role", ""))
        content = msg.get("value", msg.get("content", ""))
        if role and content:
            parts.append(f"{role}:\n{content}")
    return "\n\n".join(parts) if parts else None


def format_generic_text(example):
    """Generic: use 'text' column directly"""
    return example.get("text", None)


def format_chimera(example):
    """TianHongZXY/CHIMERA: expert-level reasoning problems with long CoT"""
    problem = example.get("problem", example.get("question", ""))
    # Try multiple possible field names for the reasoning/solution
    thinking = example.get("thinking", example.get("reasoning", example.get("thought", "")))
    solution = example.get("solution", example.get("answer", example.get("reference_solution", "")))
    subject = example.get("subject", example.get("topic", ""))

    if not problem:
        return None
    text = ""
    if subject:
        text += f"Subject: {subject}\n\n"
    text += f"Problem:\n{problem}"
    if thinking:
        text += f"\n\nThinking:\n{thinking}"
    if solution:
        text += f"\n\nSolution:\n{solution}"
    return text


FORMATTERS = {
    "open-r1/OpenR1-Math-220k": format_openr1_math,
    "open-thoughts/OpenThoughts-114k": format_openthoughts,
    "AI-MO/NuminaMath-CoT": format_numina_cot,
    "bespokelabs/Bespoke-Stratos-17k": format_stratos,
    "TianHongZXY/CHIMERA": format_chimera,
}

DATASET_CONFIGS = {
    "open-thoughts/OpenThoughts-114k": "metadata",
}

# ============================================================================
# Magic prime calculation
# ============================================================================

def is_prime(n):
    if n <= 1: return False
    if n <= 3: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0: return False
        i += 6
    return True

def find_magic_prime(data_size, ctx_len):
    n_chunk = int(data_size // ctx_len) - 1
    for i in range(n_chunk, 0, -1):
        if i % 3 == 2 and is_prime(i):
            return i
    return None

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare reasoning data for RADLADS training")
    parser.add_argument("--dataset", type=str, required=True,
                        help="HuggingFace dataset path")
    parser.add_argument("--ctxlen", type=int, default=4096,
                        help="Context length for magic_prime calculation")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Tokenizer to use")
    parser.add_argument("--out", type=str, default=None,
                        help="Output prefix (default: data/<dataset_basename>)")
    parser.add_argument("--max_tokens", type=int, default=500_000_000,
                        help="Maximum tokens to process")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    dataset_name = args.dataset
    basename = dataset_name.split("/")[-1] if "/" in dataset_name else dataset_name
    out_prefix = args.out or f"data/{basename}"

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    # select formatter
    formatter = FORMATTERS.get(dataset_name, format_generic_text)
    ds_config = DATASET_CONFIGS.get(dataset_name, None)

    print(f"Dataset:   {dataset_name}")
    print(f"Config:    {ds_config}")
    print(f"Formatter: {formatter.__name__}")
    print(f"Output:    {out_prefix}.bin / .idx")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Max tokens: {args.max_tokens:,}")
    print()

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    token_dtype = np.int32 if tokenizer.vocab_size > 65536 else np.uint16
    eos_id = tokenizer.eos_token_id

    # load dataset
    print("Loading dataset...")
    if ds_config:
        dataset = load_dataset(dataset_name, ds_config, split=args.split, streaming=True)
    else:
        dataset = load_dataset(dataset_name, split=args.split, streaming=True)

    # build binidx
    builder = MMapIndexedDatasetBuilder(f"{out_prefix}.bin", dtype=token_dtype)
    skipped = 0

    pbar = tqdm(dataset, desc="Processing", unit=" examples")
    for example in pbar:
        text = formatter(example)
        if text is None or len(text.strip()) == 0:
            skipped += 1
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens.append(eos_id)
        builder.add_doc(tokens)

        if builder.token_count() >= args.max_tokens:
            break

        if builder.token_count() % 1_000_000 < 1000:
            pbar.set_postfix(tokens=f"{builder.token_count():,}", skipped=skipped)

    builder.finalize(f"{out_prefix}.idx")

    total_tokens = builder.token_count()
    total_docs = len(builder._sizes)
    print(f"\nDone! {total_tokens:,} tokens, {total_docs:,} documents, {skipped} skipped")

    # compute magic_prime
    magic_prime = find_magic_prime(total_tokens, args.ctxlen)
    if magic_prime:
        print(f"\n{'='*60}")
        print(f"  magic_prime = {magic_prime}  (for ctxlen {args.ctxlen})")
        print(f"  --my_exit_tokens {total_tokens} --magic_prime {magic_prime} --ctx_len {args.ctxlen}")
        print(f"{'='*60}")

        # save training params to a file for easy reference
        params_file = f"{out_prefix}_params.txt"
        with open(params_file, "w") as f:
            f.write(f"my_exit_tokens={total_tokens}\n")
            f.write(f"magic_prime={magic_prime}\n")
            f.write(f"ctx_len={args.ctxlen}\n")
            f.write(f"data_file={out_prefix}\n")
        print(f"  Params saved to {params_file}")
    else:
        print(f"\nWARNING: Not enough data to compute magic_prime for ctxlen {args.ctxlen}")
        print(f"  Need at least {args.ctxlen * 3} tokens, got {total_tokens}")

if __name__ == "__main__":
    main()
