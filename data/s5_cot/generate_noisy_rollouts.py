import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model import GPTConfig, GPT
from data.s5_cot.task import (
    encode,
    sample_prompt_only,
    corrupt_token,
)

@torch.no_grad()
def load_model(teacher_out_dir, device):
    ckpt_path = os.path.join(teacher_out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)

    model_args = checkpoint["model_args"]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model

@torch.no_grad()
def sample_next_token(model, idx, temperature=1.0, top_k=None):
    idx_cond = idx[:, -model.config.block_size :]
    logits, _ = model(idx_cond)
    logits = logits[:, -1, :]

    if temperature <= 0:
        return torch.argmax(logits, dim=-1)

    logits = logits / temperature

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)

@torch.no_grad()
def build_example(model, m, eta, device, temperature, top_k):
    prompt_tokens = sample_prompt_only(m=m)
    prompt_ids = encode(prompt_tokens)
    prompt_len = len(prompt_ids)

    target_len = 7 * m  # full CoT length for S5
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated = []
    for _ in range(target_len):
        next_id = sample_next_token(model, idx, temperature=temperature, top_k=top_k).item()
        corrupted_id = corrupt_token(next_id, eta)
        generated.append(corrupted_id)

        next_tok = torch.tensor([[corrupted_id]], dtype=torch.long, device=device)
        idx = torch.cat([idx, next_tok], dim=1)

    seq = prompt_ids + generated
    x = torch.tensor(seq[:-1], dtype=torch.uint8)
    y = torch.tensor(seq[1:], dtype=torch.int16)
    y[: prompt_len - 1] = -1
    return x, y

@torch.no_grad()
def make_split(model, n, m, eta, device, temperature, top_k, split_name):
    seq_len = 14 * m  # x/y length for S5 CoT
    x_all = torch.empty((n, seq_len), dtype=torch.uint8)
    y_all = torch.empty((n, seq_len), dtype=torch.int16)

    for i in range(n):
        x, y = build_example(model, m, eta, device, temperature, top_k)
        x_all[i] = x
        y_all[i] = y
        if (i + 1) % 1000 == 0 or (i + 1) == n:
            print(f"{split_name}: {i+1}/{n}")

    return x_all, y_all

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_out_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--m", type=int, default=21)
    parser.add_argument("--n_train", type=int, default=50000)
    parser.add_argument("--n_val", type=int, default=5000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if "cuda" in args.device and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = load_model(args.teacher_out_dir, args.device)

    train_x, train_y = make_split(
        model=model,
        n=args.n_train,
        m=args.m,
        eta=args.eta,
        device=args.device,
        temperature=args.temperature,
        top_k=args.top_k,
        split_name="train",
    )

    val_x, val_y = make_split(
        model=model,
        n=args.n_val,
        m=args.m,
        eta=args.eta,
        device=args.device,
        temperature=args.temperature,
        top_k=args.top_k,
        split_name="val",
    )

    torch.save(train_x, os.path.join(args.save_dir, "train_x.pt"))
    torch.save(train_y, os.path.join(args.save_dir, "train_y.pt"))
    torch.save(val_x, os.path.join(args.save_dir, "val_x.pt"))
    torch.save(val_y, os.path.join(args.save_dir, "val_y.pt"))

    meta = {
        "eta": args.eta,
        "m": args.m,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "teacher_out_dir": args.teacher_out_dir,
    }
    with open(os.path.join(args.save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"saved dataset to {args.save_dir}")

if __name__ == "__main__":
    main()