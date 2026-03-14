"""
Chatbot SFT 数据集预处理脚本
将 HuggingFace 上的 chatbot/instruction-following 数据集转换为 RADLADS 训练所需的 binidx 格式

支持的数据集:
    1. Open-Orca/SlimOrca          - 518k 高质量指令数据 (GPT-4 生成, ShareGPT 格式)
    2. HuggingFaceH4/ultrachat_200k - 200k 多轮对话 (合成, 高质量)
    3. teknium/OpenHermes-2.5       - 1M 指令数据 (多源混合, ShareGPT 格式)
    4. mlabonne/WizardLM_evol_instruct_70k - 70k 进化指令 (Alpaca 格式)
    5. m-a-p/CodeFeedback-Filtered-Instruction - 157k 代码指令

用法:
    python prepare_chatbot_data.py --dataset Open-Orca/SlimOrca --ctxlen 4096
    python prepare_chatbot_data.py --dataset HuggingFaceH4/ultrachat_200k --ctxlen 4096
    python prepare_chatbot_data.py --dataset teknium/OpenHermes-2.5 --ctxlen 4096
    python prepare_chatbot_data.py --dataset mlabonne/WizardLM_evol_instruct_70k --ctxlen 4096
    python prepare_chatbot_data.py --dataset m-a-p/CodeFeedback-Filtered-Instruction --ctxlen 4096
"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

from src.binidx import MMapIndexedDataset

# ============================================================================
# binidx builder (from prepare_reasoning_data.py)
# ============================================================================

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
# Chatbot dataset formatters - 使用 chat template 格式化
# ============================================================================

def format_slimorca(example, tokenizer):
    """Open-Orca/SlimOrca: ShareGPT 格式 (conversations 字段)"""
    conversations = example.get("conversations", [])
    if not conversations:
        return None
    messages = []
    for msg in conversations:
        role_map = {"system": "system", "human": "user", "gpt": "assistant"}
        role = role_map.get(msg.get("from", ""), msg.get("from", ""))
        content = msg.get("value", "")
        if role and content:
            messages.append({"role": role, "content": content})
    if not messages:
        return None
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        return None


def format_ultrachat(example, tokenizer):
    """HuggingFaceH4/ultrachat_200k: messages 字段"""
    messages = example.get("messages", [])
    if not messages:
        return None
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        return None


def format_openhermes(example, tokenizer):
    """teknium/OpenHermes-2.5: ShareGPT 格式 (conversations 字段)"""
    conversations = example.get("conversations", [])
    if not conversations:
        return None
    messages = []
    for msg in conversations:
        role_map = {"system": "system", "human": "user", "gpt": "assistant"}
        role = role_map.get(msg.get("from", ""), msg.get("from", ""))
        content = msg.get("value", "")
        if role and content:
            messages.append({"role": role, "content": content})
    if not messages:
        return None
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        return None


def format_wizardlm(example, tokenizer):
    """mlabonne/WizardLM_evol_instruct_70k: Alpaca 格式 (instruction + output)"""
    instruction = example.get("instruction", "")
    output = example.get("output", "")
    if not instruction or not output:
        return None
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        return None


def format_codefeedback(example, tokenizer):
    """m-a-p/CodeFeedback-Filtered-Instruction: query + answer"""
    query = example.get("query", "")
    answer = example.get("answer", "")
    if not query or not answer:
        return None
    messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        return None


def format_generic_messages(example, tokenizer):
    """通用: 尝试 messages 字段, 然后 conversations 字段, 最后 text 字段"""
    messages = example.get("messages", None)
    if messages:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass

    conversations = example.get("conversations", [])
    if conversations:
        msgs = []
        for msg in conversations:
            role_map = {"system": "system", "human": "user", "gpt": "assistant"}
            role = role_map.get(msg.get("from", ""), msg.get("role", ""))
            content = msg.get("value", msg.get("content", ""))
            if role and content:
                msgs.append({"role": role, "content": content})
        if msgs:
            try:
                return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            except Exception:
                pass

    text = example.get("text", None)
    if text:
        return text

    return None


FORMATTERS = {
    "Open-Orca/SlimOrca": format_slimorca,
    "HuggingFaceH4/ultrachat_200k": format_ultrachat,
    "teknium/OpenHermes-2.5": format_openhermes,
    "mlabonne/WizardLM_evol_instruct_70k": format_wizardlm,
    "m-a-p/CodeFeedback-Filtered-Instruction": format_codefeedback,
}

DATASET_CONFIGS = {
    "HuggingFaceH4/ultrachat_200k": "default",
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
    parser = argparse.ArgumentParser(description="Prepare chatbot SFT data for RADLADS training")
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
    formatter = FORMATTERS.get(dataset_name, format_generic_messages)
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
        text = formatter(example, tokenizer)
        if text is None or len(text.strip()) == 0:
            skipped += 1
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        # 确保以 eos 结尾
        if tokens[-1] != eos_id:
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
        print(f"  data_file={out_prefix}")
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
