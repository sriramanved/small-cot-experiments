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
from pathlib import Path
from typing import Any, Mapping

import torch
from transformers import GPT2Config, GPT2LMHeadModel


UNWANTED_PREFIX = "_orig_mod."
TRANSPOSED_WEIGHTS = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)
IGNORED_NANOGPT_KEYS = (".attn.bias",)
DTYPE_LOOKUP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_checkpoint_path(checkpoint_or_dir: str | Path) -> Path:
    path = Path(checkpoint_or_dir)
    if path.is_dir():
        path = path / "ckpt.pt"
    if not path.exists():
        raise FileNotFoundError(f"Could not find checkpoint at {path}")
    return path


def load_nanogpt_checkpoint(
    checkpoint_or_dir: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    ckpt_path = resolve_checkpoint_path(checkpoint_or_dir)
    return torch.load(ckpt_path, map_location=map_location)


def clean_nanogpt_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith(UNWANTED_PREFIX):
            key = key[len(UNWANTED_PREFIX):]
        if any(key.endswith(suffix) for suffix in IGNORED_NANOGPT_KEYS):
            continue
        cleaned[key] = value
    return cleaned


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

    return GPT2Config(
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
