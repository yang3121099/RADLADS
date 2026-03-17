########################################################################################################
# Convert RADLADS training checkpoint to HuggingFace model directory format.
#
# Usage:
#   # Pure Qwen2 model (vllm compatible):
#   python convert_to_hf.py -c configs/qwen7b.yaml --path /path/to/checkpoint.safetensors --out /path/to/hf_model
#
#   # RWKV6 hybrid model (trust_remote_code required):
#   python convert_to_hf.py -c configs/qwen7b.yaml -c configs/qwerky6.yaml --path /path/to/checkpoint.safetensors --out /path/to/hf_model
#
#   # Custom tokenizer:
#   python convert_to_hf.py -c configs/qwen7b.yaml --path ckpt.safetensors --out hf_model --tokenizer Qwen/Qwen2.5-7B
#
########################################################################################################

import os, sys, json, shutil
import typing
from dataclasses import dataclass

import torch
from safetensors.torch import load_file
from huggingface_hub import save_torch_state_dict
from transformers import AutoTokenizer

from configs import parse_cmdline_configs, Model_Config, Transformer_Config

########################################################################################################

# Default tokenizer lookup by vocab_size
TOKENIZER_BY_VOCAB = {
    151936: 'Qwen/Qwen2-0.5B',
    152064: 'Qwen/Qwen2.5-7B',
}

@dataclass(kw_only=True)
class CLI_Config:
    path: str                          # input checkpoint path (.pth or .safetensors)
    out: str                           # output HuggingFace model directory
    tokenizer: str = ''                # tokenizer name/path (auto-detected from vocab_size if empty)
    train: typing.Any = None
    model: Model_Config

########################################################################################################

def detect_model_type(tmix: str):
    """Detect model architecture type from tmix config value."""
    if tmix.startswith('qwen2') or tmix == '':
        return 'qwen2'
    if 'rwkv6' in tmix or 'qwerky6' in tmix:
        return 'rwkv6'
    if 'rwkv7' in tmix or 'qwerky7' in tmix:
        return 'rwkv7'
    # Default: treat unknown tmix as needing trust_remote_code
    return 'unknown'

def build_qwen2_config(model_cfg):
    """Build a standard HuggingFace Qwen2Config dict."""
    from transformers import Qwen2Config

    head_size = getattr(model_cfg, 'head_size', 128)
    num_attention_heads = model_cfg.n_embd // head_size
    num_kv_heads = getattr(model_cfg, 'num_key_value_heads', 0) or num_attention_heads
    rope_theta = 10000.0
    if hasattr(model_cfg, 'rope') and model_cfg.rope is not None:
        rope_theta = model_cfg.rope.base

    hf_config = Qwen2Config(
        vocab_size=model_cfg.vocab_size,
        hidden_size=model_cfg.n_embd,
        intermediate_size=model_cfg.dim_ffn,
        num_hidden_layers=model_cfg.n_layer,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_kv_heads,
        max_position_embeddings=max(model_cfg.ctx_len, 32768),
        rms_norm_eps=model_cfg.rms_norm_eps,
        rope_theta=rope_theta,
        tie_word_embeddings=False,
        use_sliding_window=False,
    )
    return hf_config

def build_rwkv6_config(model_cfg):
    """Build RWKV6Qwen2Config dict."""
    from rwkv6qwen2.configuration_rwkv6qwen2 import RWKV6Qwen2Config

    head_size = getattr(model_cfg, 'head_size', 128)
    num_attention_heads = model_cfg.n_embd // head_size
    num_kv_heads = getattr(model_cfg, 'num_key_value_heads', 0) or num_attention_heads
    rope_theta = 10000.0
    use_rope = False
    if hasattr(model_cfg, 'rope') and model_cfg.rope is not None:
        rope_theta = model_cfg.rope.base
        use_rope = True

    hf_config = RWKV6Qwen2Config(
        vocab_size=model_cfg.vocab_size,
        hidden_size=model_cfg.n_embd,
        intermediate_size=model_cfg.dim_ffn,
        num_hidden_layers=model_cfg.n_layer,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_kv_heads,
        lora_rank_tokenshift=getattr(model_cfg, 'lora_rank_tokenshift', None),
        lora_rank_decay=getattr(model_cfg, 'lora_rank_decay', None),
        max_position_embeddings=max(model_cfg.ctx_len, 32768),
        rms_norm_eps=model_cfg.rms_norm_eps,
        use_rope=use_rope,
        rope_theta=rope_theta,
        tie_word_embeddings=False,
        gate_rank_type=getattr(model_cfg, 'gate_rank_type', 1),
        lora_rank_gate=getattr(model_cfg, 'lora_rank_gate', None) or None,
        balance_state=bool(getattr(model_cfg, 'balance_state', 1)),
        groupnorm_att=bool(getattr(model_cfg, 'groupnorm_att', 0)),
        use_tokenshift=bool(getattr(model_cfg, 'use_tokenshift', 1)),
    )
    return hf_config

def build_rwkv7_config(model_cfg):
    """Build RWKV7Qwen2Config dict."""
    from rwkv7qwen2.configuration_rwkv7qwen2 import RWKV7Qwen2Config

    head_size = getattr(model_cfg, 'head_size', 128)
    num_attention_heads = model_cfg.n_embd // head_size
    num_kv_heads = getattr(model_cfg, 'num_key_value_heads', 0) or num_attention_heads
    rope_theta = 10000.0
    use_rope = False
    if hasattr(model_cfg, 'rope') and model_cfg.rope is not None:
        rope_theta = model_cfg.rope.base
        use_rope = True

    hf_config = RWKV7Qwen2Config(
        vocab_size=model_cfg.vocab_size,
        hidden_size=model_cfg.n_embd,
        intermediate_size=model_cfg.dim_ffn,
        num_hidden_layers=model_cfg.n_layer,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_kv_heads,
        lora_rank_decay=getattr(model_cfg, 'lora_rank_decay', None),
        lora_rank_iclr=getattr(model_cfg, 'lora_rank_iclr', None),
        lora_rank_value_residual_mix=getattr(model_cfg, 'lora_rank_value_residual_mix', None),
        lora_rank_gate=getattr(model_cfg, 'lora_rank_gate', None) or None,
        max_position_embeddings=max(model_cfg.ctx_len, 32768),
        rms_norm_eps=model_cfg.rms_norm_eps,
        use_rope=use_rope,
        rope_theta=rope_theta,
        tie_word_embeddings=False,
        gate_rank_type=getattr(model_cfg, 'gate_rank_type', 2),
        balance_state=bool(getattr(model_cfg, 'balance_state', 0)),
        groupnorm_att=bool(getattr(model_cfg, 'groupnorm_att', 1)),
        use_tokenshift=bool(getattr(model_cfg, 'use_tokenshift', 0)),
    )
    return hf_config

def copy_model_code(model_type: str, out_dir: str):
    """Copy custom modeling files for trust_remote_code support."""
    src_dir = os.path.join(os.path.dirname(__file__), f'{model_type}qwen2')
    if not os.path.isdir(src_dir):
        print(f"WARNING: Custom model code directory not found: {src_dir}")
        return

    for fname in os.listdir(src_dir):
        if fname.endswith('.py'):
            shutil.copy2(os.path.join(src_dir, fname), os.path.join(out_dir, fname))
            print(f'  Copied {fname}')

def patch_auto_map(out_dir: str, model_type: str):
    """Add auto_map to config.json for trust_remote_code."""
    config_path = os.path.join(out_dir, 'config.json')
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    type_prefix = model_type.upper() + 'Qwen2'
    cfg_class = f'configuration_{model_type}qwen2.{type_prefix}Config'
    model_class = f'modeling_{model_type}qwen2.{type_prefix}ForCausalLM'

    cfg['auto_map'] = {
        'AutoConfig': cfg_class,
        'AutoModelForCausalLM': model_class,
    }

    with open(config_path, 'w') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

########################################################################################################

def main():
    config, errors = parse_cmdline_configs(sys.argv[1:], CLI_Config)
    if errors:
        print(errors)
        exit(1)

    model_cfg = config.model

    # ---- Load checkpoint ----
    print(f'Loading checkpoint: {config.path}')
    if config.path.lower().endswith('.safetensors'):
        state_dict = load_file(config.path, device='cpu')
    else:
        state_dict = torch.load(config.path, map_location='cpu', weights_only=True)

    # Handle tied embeddings
    if 'lm_head.weight' not in state_dict:
        print('  lm_head.weight not found, tying to model.embed_tokens.weight')
        state_dict['lm_head.weight'] = state_dict['model.embed_tokens.weight']

    # ---- Detect model type ----
    tmix = model_cfg.tmix
    model_type = detect_model_type(tmix)
    print(f'Detected model type: {model_type} (tmix={tmix})')

    # ---- Create output directory ----
    os.makedirs(config.out, exist_ok=True)

    # ---- Build & save HF config ----
    if model_type == 'qwen2':
        hf_config = build_qwen2_config(model_cfg)
    elif model_type == 'rwkv6':
        hf_config = build_rwkv6_config(model_cfg)
    elif model_type == 'rwkv7':
        hf_config = build_rwkv7_config(model_cfg)
    else:
        print(f'WARNING: Unknown model type "{tmix}", attempting Qwen2 config as fallback')
        hf_config = build_qwen2_config(model_cfg)

    hf_config.save_pretrained(config.out)
    print(f'Saved config.json ({hf_config.model_type})')

    # ---- Copy custom model code for trust_remote_code (RWKV models) ----
    if model_type in ('rwkv6', 'rwkv7'):
        print(f'Copying {model_type}qwen2 model code for trust_remote_code...')
        copy_model_code(model_type, config.out)
        patch_auto_map(config.out, model_type)

    # ---- Save weights as sharded safetensors ----
    print(f'Saving model weights to {config.out} ...')
    save_torch_state_dict(state_dict, config.out)

    # ---- Save tokenizer ----
    tokenizer_name = config.tokenizer
    if not tokenizer_name:
        tokenizer_name = TOKENIZER_BY_VOCAB.get(model_cfg.vocab_size, 'Qwen/Qwen2-0.5B')
        print(f'Auto-detected tokenizer: {tokenizer_name} (vocab_size={model_cfg.vocab_size})')
    print(f'Saving tokenizer from {tokenizer_name} ...')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.save_pretrained(config.out)

    # ---- Summary ----
    print()
    print('=' * 60)
    print(f'Conversion complete: {config.out}')
    print(f'  Model type:  {hf_config.model_type}')
    print(f'  Layers:      {hf_config.num_hidden_layers}')
    print(f'  Hidden size: {hf_config.hidden_size}')
    print(f'  Vocab size:  {hf_config.vocab_size}')
    if model_type == 'qwen2':
        print()
        print('  This is a standard Qwen2 model. You can use it with vllm:')
        print(f'    python run_lm_eval_vllm.py --model {config.out} --tasks lambada_openai')
    else:
        print()
        print(f'  This is a {model_type}qwen2 model (requires trust_remote_code).')
        print(f'  Use with lm_eval hf backend:')
        print(f'    python run_lm_eval_vllm.py --model {config.out} --tasks lambada_openai --backend hf')
    print('=' * 60)

if __name__ == '__main__':
    main()
