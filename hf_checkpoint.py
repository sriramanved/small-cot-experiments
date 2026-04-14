"""
Utilities for loading nanoGPT checkpoints as Hugging Face GPT-2 models.

Example:
    from hf_checkpoint import load_nanogpt_checkpoint_as_hf

    model = load_nanogpt_checkpoint_as_hf("out-s5-clean-offline-bc")
    generated = model.generate(input_ids, max_new_tokens=32)

This utility converts the repo's saved `ckpt.pt` format into a
`transformers.GPT2LMHeadModel`, and can optionally export a
`save_pretrained(...)` directory for later reuse.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D

from nanogpt_checkpoint import (
    load_nanogpt_checkpoint,
    normalize_nanogpt_state_dict,
    resolve_checkpoint_path,
)
from torch_dtypes import DTYPE_LOOKUP

TRANSPOSED_WEIGHTS = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)
IGNORED_NANOGPT_KEYS = (".attn.bias",)
def clean_nanogpt_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = normalize_nanogpt_state_dict(state_dict)
    return {
        key: value
        for key, value in cleaned.items()
        if not any(key.endswith(suffix) for suffix in IGNORED_NANOGPT_KEYS)
    }


def nanogpt_model_args_to_hf_config(
    model_args: Mapping[str, Any],
    *,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
) -> GPT2Config:
    dropout = float(model_args.get("dropout", 0.0))
    n_embd = int(model_args["n_embd"])
    block_size = int(model_args["block_size"])

    config = GPT2Config(
        vocab_size=int(model_args["vocab_size"]),
        n_positions=block_size,
        n_ctx=block_size,
        n_embd=n_embd,
        n_layer=int(model_args["n_layer"]),
        n_head=int(model_args["n_head"]),
        n_inner=4 * n_embd,
        activation_function="gelu",
        resid_pdrop=dropout,
        embd_pdrop=dropout,
        attn_pdrop=dropout,
        layer_norm_epsilon=1e-5,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        use_cache=True,
    )
    config.loss_type = "ForCausalLM"
    return config


def init_hf_model_like_nanogpt(
    hf_model: GPT2LMHeadModel,
    *,
    has_bias: bool,
) -> None:
    for module in hf_model.modules():
        if isinstance(module, (nn.Linear, Conv1D)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    for name, param in hf_model.named_parameters():
        if name.endswith("c_proj.weight"):
            nn.init.normal_(
                param,
                mean=0.0,
                std=0.02 / math.sqrt(2 * hf_model.config.n_layer),
            )

    apply_nanogpt_bias_policy(
        hf_model,
        has_bias=has_bias,
    )
    hf_model.tie_weights()


def set_hf_causal_lm_loss(hf_model: GPT2LMHeadModel) -> None:
    hf_model.loss_type = "ForCausalLM"


def apply_nanogpt_bias_policy(
    hf_model: GPT2LMHeadModel,
    *,
    has_bias: bool,
) -> None:
    if not has_bias:
        for name, param in hf_model.named_parameters():
            if name.endswith(".bias"):
                with torch.no_grad():
                    param.zero_()
                param.requires_grad = False
    else:
        for name, param in hf_model.named_parameters():
            if name.endswith(".bias"):
                param.requires_grad = True


def build_hf_model_from_nanogpt_args(
    model_args: Mapping[str, Any],
    *,
    device: str | torch.device | None = None,
    torch_dtype: torch.dtype | None = None,
    eval_mode: bool = False,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
) -> GPT2LMHeadModel:
    hf_config = nanogpt_model_args_to_hf_config(
        model_args,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    hf_config.nanogpt_bias = bool(model_args.get("bias", True))
    hf_model = GPT2LMHeadModel(hf_config)
    set_hf_causal_lm_loss(hf_model)
    init_hf_model_like_nanogpt(
        hf_model,
        has_bias=bool(model_args.get("bias", True)),
    )

    if eval_mode:
        hf_model.eval()

    if torch_dtype is not None and device is not None:
        hf_model.to(device=device, dtype=torch_dtype)
    elif torch_dtype is not None:
        hf_model.to(dtype=torch_dtype)
    elif device is not None:
        hf_model.to(device=device)

    return hf_model


def convert_nanogpt_state_dict_to_hf(
    nanogpt_state_dict: Mapping[str, torch.Tensor],
    hf_model: GPT2LMHeadModel,
    *,
    has_bias: bool,
) -> dict[str, torch.Tensor]:
    hf_reference = hf_model.state_dict()
    converted: dict[str, torch.Tensor] = {}

    unexpected_keys = [
        key for key in nanogpt_state_dict.keys() if key not in hf_reference
    ]
    if unexpected_keys:
        raise KeyError(
            "Checkpoint has keys that do not match Hugging Face GPT-2: "
            + ", ".join(unexpected_keys)
        )

    for key, value in nanogpt_state_dict.items():
        target = hf_reference[key]
        if any(key.endswith(suffix) for suffix in TRANSPOSED_WEIGHTS):
            value = value.t()
        if value.shape != target.shape:
            raise ValueError(
                f"Shape mismatch for {key}: checkpoint {tuple(value.shape)} "
                f"!= HF {tuple(target.shape)}"
            )
        converted[key] = value.to(dtype=target.dtype)

    missing_keys = [key for key in hf_reference.keys() if key not in converted]
    for key in missing_keys:
        if not has_bias and key.endswith(".bias"):
            converted[key] = torch.zeros_like(hf_reference[key])
            continue
        raise KeyError(
            "Converted state dict is missing required Hugging Face key "
            f"{key}. If this is intentional, extend the converter for it."
        )

    return converted


def load_nanogpt_checkpoint_as_hf(
    checkpoint_or_dir: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    device: str | torch.device | None = None,
    torch_dtype: torch.dtype | None = None,
    eval_mode: bool = True,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
) -> GPT2LMHeadModel:
    checkpoint = load_nanogpt_checkpoint(checkpoint_or_dir, map_location=map_location)
    model_args = checkpoint["model_args"]

    hf_config = nanogpt_model_args_to_hf_config(
        model_args,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    hf_model = GPT2LMHeadModel(hf_config)
    set_hf_causal_lm_loss(hf_model)

    cleaned_state_dict = clean_nanogpt_state_dict(checkpoint["model"])
    converted_state_dict = convert_nanogpt_state_dict_to_hf(
        cleaned_state_dict,
        hf_model,
        has_bias=bool(model_args["bias"]),
    )
    hf_model.load_state_dict(converted_state_dict, strict=True)
    hf_model.tie_weights()

    if eval_mode:
        hf_model.eval()

    if torch_dtype is not None and device is not None:
        hf_model.to(device=device, dtype=torch_dtype)
    elif torch_dtype is not None:
        hf_model.to(dtype=torch_dtype)
    elif device is not None:
        hf_model.to(device=device)

    return hf_model


def export_nanogpt_checkpoint_to_hf(
    checkpoint_or_dir: str | Path,
    save_dir: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    device: str | torch.device | None = None,
    torch_dtype: torch.dtype | None = None,
    safe_serialization: bool = False,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
) -> Path:
    model = load_nanogpt_checkpoint_as_hf(
        checkpoint_or_dir,
        map_location=map_location,
        device=device,
        torch_dtype=torch_dtype,
        eval_mode=True,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path, safe_serialization=safe_serialization)
    return save_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load or export a nanoGPT checkpoint as a Hugging Face GPT2LMHeadModel."
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to ckpt.pt or an out_dir containing ckpt.pt",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Optional output directory for save_pretrained(...).",
    )
    parser.add_argument(
        "--map_location",
        type=str,
        default="cpu",
        help="torch.load map_location for the checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device to move the Hugging Face model onto, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPE_LOOKUP),
        default=None,
        help="Optional dtype to cast the Hugging Face model to after loading.",
    )
    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        help="Save weights as safetensors instead of pytorch_model.bin.",
    )
    parser.add_argument(
        "--bos_token_id",
        type=int,
        default=None,
        help="Optional BOS token id to store in the Hugging Face config.",
    )
    parser.add_argument(
        "--eos_token_id",
        type=int,
        default=None,
        help="Optional EOS token id to store in the Hugging Face config.",
    )
    parser.add_argument(
        "--pad_token_id",
        type=int,
        default=None,
        help="Optional PAD token id to store in the Hugging Face config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch_dtype = DTYPE_LOOKUP.get(args.dtype)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)

    model = load_nanogpt_checkpoint_as_hf(
        checkpoint_path,
        map_location=args.map_location,
        device=args.device,
        torch_dtype=torch_dtype,
        eval_mode=True,
        bos_token_id=args.bos_token_id,
        eos_token_id=args.eos_token_id,
        pad_token_id=args.pad_token_id,
    )

    print(f"Loaded Hugging Face model from {checkpoint_path}")
    print(model.config)

    if args.save_dir is not None:
        save_path = Path(args.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_path, safe_serialization=args.safe_serialization)
        print(f"Saved Hugging Face model to {save_path}")


if __name__ == "__main__":
    main()
