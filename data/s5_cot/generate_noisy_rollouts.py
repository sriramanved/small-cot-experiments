import argparse
import os
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.s5_cot.offline_render import DTYPE_LOOKUP, render_offline_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render an offline S5 dataset from a fixed prompt bank using a clean "
            "teacher checkpoint. Use eta=0 for the clean offline BC sweep."
        )
    )
    parser.add_argument(
        "--teacher_checkpoint",
        "--teacher_out_dir",
        dest="teacher_checkpoint",
        type=str,
        required=True,
        help="Path to ckpt.pt or an out_dir containing ckpt.pt.",
    )
    parser.add_argument("--prompt_bank_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--subset_size", type=int, required=True)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--gen_batch_size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=sorted(DTYPE_LOOKUP), default=None)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if "cuda" in args.device and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    render_offline_dataset(
        teacher_checkpoint=args.teacher_checkpoint,
        prompt_bank_dir=args.prompt_bank_dir,
        save_dir=args.save_dir,
        subset_size=args.subset_size,
        eta=args.eta,
        gen_batch_size=args.gen_batch_size,
        device=args.device,
        dtype_name=args.dtype,
        seed=args.seed,
    )
    print(f"saved dataset to {args.save_dir}")


if __name__ == "__main__":
    main()
