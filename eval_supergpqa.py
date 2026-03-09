"""
SuperGPQA 测评脚本 (m-a-p/SuperGPQA)
用法:
    python eval_supergpqa.py \
        -c configs/qwen7b.yaml -c configs/qwerky7.yaml \
        --model.attention_type rwkv7_fla_fused_recurrent \
        --model.ctx_len 4096 --precision bf16 \
        --path out/xxx/rwkv-final.pth \
        --bsz 4 --limit 500
"""
import os, sys, json, re
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from dataclasses import dataclass
import typing

from configs import parse_cmdline_configs, Model_Config

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1'

@dataclass(kw_only=True)
class CLI_Config:
    path: str
    precision: int | str = 'bf16'
    bsz: int = 4
    limit: int | None = None
    train: typing.Any = None
    model: Model_Config

config, errors = parse_cmdline_configs(sys.argv[1:], CLI_Config)
if errors != '':
    print(errors)
    exit()
config.train = None

os.environ["RWKV_MODEL_TYPE"] = config.model.tmix
os.environ["RWKV_CTXLEN"] = str(config.model.ctx_len)
os.environ["RWKV_HEAD_SIZE_A"] = str(config.model.head_size)
attention_type = str(config.model.attention_type)
if attention_type == 'rwkv7':
    attention_type = 'rwkv7_fla_fused_recurrent'
os.environ["RWKV_ATTENTION_TYPE"] = attention_type

from pydoc import locate
from safetensors.torch import load_file

# Load model
print(f'Loading model - {config.path}')
classname = config.model.classname
if config.path.lower().endswith('.safetensors'):
    load_dict = load_file(config.path)
else:
    load_dict = torch.load(config.path, mmap=True)
if (classname.startswith('qwen2') or config.model.tmix.startswith('qwen2')) and config.model.n_embd < 3584:
    load_dict['lm_head.weight'] = load_dict['model.embed_tokens.weight']

with torch.device('meta'):
    if classname != '':
        model_classpath = f'models.{classname}.Model_{classname}'
        model_factory = locate(model_classpath)
        model = model_factory(config)
    else:
        from src.model import Transformer
        model = Transformer(config)

if hasattr(model, 'configure_model'):
    model.configure_model()
model.load_state_dict(load_dict, assign=True, strict=False)

match config.precision:
    case 32 | '32': dtype = torch.float32
    case 16 | '16': dtype = torch.float16
    case 'bf16': dtype = torch.bfloat16
    case _:
        print("Bad precision"); exit()

device = 'cuda'
model = model.to(device=device, dtype=dtype)
model.eval()

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')

# Load SuperGPQA
print("Loading SuperGPQA dataset...")
dataset = load_dataset("m-a-p/SuperGPQA", split="train")
if config.limit:
    dataset = dataset.select(range(min(config.limit, len(dataset))))
print(f"Evaluating on {len(dataset)} questions")

def format_question(item):
    """Format a SuperGPQA question into a prompt with options."""
    question = item.get("question", "")
    options = item.get("options", [])

    prompt = f"Question: {question}\n\nOptions:\n"
    for i, opt in enumerate(options):
        letter = chr(ord('A') + i)
        prompt += f"  {letter}. {opt}\n"
    prompt += "\nAnswer:"
    return prompt

def extract_answer(logits, option_tokens):
    """Extract answer from logits by comparing option token probabilities."""
    last_logits = logits[:, -1, :]  # [B, vocab]
    probs = F.softmax(last_logits, dim=-1)

    option_probs = []
    for tokens in option_tokens:
        # Take the max prob among all tokens that represent this option
        p = max(probs[0, t].item() for t in tokens)
        option_probs.append(p)

    return chr(ord('A') + np.argmax(option_probs))

# Precompute option letter tokens
option_letter_tokens = {}
for letter in "ABCDEFGHIJ":
    tokens = set()
    for variant in [letter, f" {letter}", f"{letter}.", f" {letter}."]:
        toks = tokenizer.encode(variant, add_special_tokens=False)
        tokens.update(toks)
    # Also try single token
    single = tokenizer.encode(letter, add_special_tokens=False)
    tokens.update(single)
    option_letter_tokens[letter] = list(tokens)

# Evaluate
correct = 0
total = 0
results_by_discipline = defaultdict(lambda: {"correct": 0, "total": 0})
results_by_difficulty = defaultdict(lambda: {"correct": 0, "total": 0})

with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=dtype):
    for i in tqdm(range(0, len(dataset), config.bsz), desc="SuperGPQA"):
        batch_items = [dataset[j] for j in range(i, min(i + config.bsz, len(dataset)))]

        # Format and tokenize
        prompts = [format_question(item) for item in batch_items]
        encoded = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]

        # Pad to same length
        maxlen = max(len(e) for e in encoded)
        maxlen = min(maxlen, config.model.ctx_len)
        maxlen = (maxlen + 7) // 8 * 8

        batch_inputs = []
        for e in encoded:
            e = e[-maxlen:]  # truncate from left if too long
            padded = [0] * (maxlen - len(e)) + e
            batch_inputs.append(torch.tensor(padded, dtype=torch.long, device=device))
        batch_inputs = torch.stack(batch_inputs)

        # Forward
        results = model.forward(batch_inputs, None)
        if isinstance(results, tuple):
            logits = results[0]
        elif isinstance(results, torch.Tensor):
            logits = results
        else:
            logits = results.logits

        # Score each item in batch
        for b, item in enumerate(batch_items):
            answer_letter = item.get("answer_letter", "")
            options = item.get("options", [])
            n_options = len(options)
            discipline = item.get("discipline", "unknown")
            difficulty = item.get("difficulty", "unknown")

            # Get prediction from logits
            item_logits = logits[b:b+1]
            # Find the actual end position (before padding)
            enc_len = len(encoded[b])
            actual_end = min(enc_len, maxlen)
            item_logits_at_end = logits[b:b+1, actual_end-1:actual_end, :]

            probs = F.softmax(item_logits_at_end.squeeze(0).squeeze(0), dim=-1)

            # Compare probabilities for each option letter
            best_letter = "A"
            best_prob = -1
            for idx in range(n_options):
                letter = chr(ord('A') + idx)
                if letter in option_letter_tokens:
                    p = max(probs[t].item() for t in option_letter_tokens[letter])
                    if p > best_prob:
                        best_prob = p
                        best_letter = letter

            is_correct = (best_letter == answer_letter)
            if is_correct:
                correct += 1
            total += 1

            results_by_discipline[discipline]["total"] += 1
            results_by_discipline[discipline]["correct"] += int(is_correct)
            results_by_difficulty[difficulty]["total"] += 1
            results_by_difficulty[difficulty]["correct"] += int(is_correct)

# Print results
print(f"\n{'='*60}")
print(f"SuperGPQA Results: {config.path}")
print(f"{'='*60}")
print(f"Overall Accuracy: {correct}/{total} = {correct/total*100:.2f}%")

print(f"\nBy Difficulty:")
for diff in sorted(results_by_difficulty.keys()):
    r = results_by_difficulty[diff]
    acc = r['correct'] / r['total'] * 100 if r['total'] > 0 else 0
    print(f"  {diff:10s}: {r['correct']:4d}/{r['total']:4d} = {acc:.2f}%")

print(f"\nBy Discipline:")
for disc in sorted(results_by_discipline.keys()):
    r = results_by_discipline[disc]
    acc = r['correct'] / r['total'] * 100 if r['total'] > 0 else 0
    print(f"  {disc:30s}: {r['correct']:4d}/{r['total']:4d} = {acc:.2f}%")
