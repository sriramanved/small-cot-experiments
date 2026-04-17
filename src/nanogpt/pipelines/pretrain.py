from __future__ import annotations

from pathlib import Path

from nanogpt.config_schema import AppConfig
from nanogpt.utils.repo import repo_root, script_path


def _bool_string(value: bool) -> str:
    return "True" if value else "False"


def _add_kv(args: list[str], key: str, value) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        args.append(f"--{key}={_bool_string(value)}")
        return
    if value == "":
        return
    args.append(f"--{key}={value}")


def _torchrun_bin(python_bin: str) -> str:
    candidate = Path(python_bin).resolve().parent / "torchrun"
    if candidate.exists():
        return str(candidate)
    return "torchrun"


def build_command(cfg: AppConfig) -> list[str]:
    torchrun = cfg.runtime.torchrun
    if torchrun.nproc_per_node > 1 or torchrun.nnodes > 1:
        command = [_torchrun_bin(cfg.runtime.python_bin)]
        if torchrun.standalone and torchrun.nnodes == 1:
            command.append("--standalone")
        command.extend(
            [
                f"--nproc_per_node={torchrun.nproc_per_node}",
                f"--nnodes={torchrun.nnodes}",
                f"--node_rank={torchrun.node_rank}",
                f"--master_addr={torchrun.master_addr}",
                f"--master_port={torchrun.master_port}",
                str(script_path("train.py")),
            ]
        )
    else:
        command = [cfg.runtime.python_bin, str(script_path("train.py"))]

    _add_kv(command, "out_dir", cfg.run.out_dir)
    _add_kv(command, "dataset", cfg.task.dataset)
    _add_kv(command, "s5_mode", cfg.task.s5_mode)
    _add_kv(command, "s5_m", cfg.task.s5_m)
    _add_kv(command, "modadd_p", cfg.task.modadd_p)
    _add_kv(command, "modadd_m", cfg.task.modadd_m)

    for key in ("n_layer", "n_head", "n_embd", "dropout", "bias", "block_size"):
        _add_kv(command, key, getattr(cfg.model, key))

    for key in (
        "eval_interval",
        "log_interval",
        "eval_iters",
        "eval_only",
        "always_save_checkpoint",
        "init_from",
        "init_from_ckpt",
        "continue_from_subset_size",
        "gradient_accumulation_steps",
        "batch_size",
        "learning_rate",
        "max_iters",
        "weight_decay",
        "beta1",
        "beta2",
        "grad_clip",
        "decay_lr",
        "warmup_iters",
        "lr_decay_iters",
        "min_lr",
        "save_every",
        "offline_single_epoch",
        "offline_eval_full",
        "offline_train_subset_size",
        "offline_train_shuffle",
        "offline_target_type",
        "final_eval_on_exit",
        "s5_eval_metrics",
        "s5_eval_clean_train_loss",
        "modadd_eval_metrics",
        "modadd_eval_clean_train_loss",
        "s5_eval_n",
        "s5_eval_batch_size",
        "s5_eval_seed",
    ):
        _add_kv(command, key, getattr(cfg.optim, key))

    _add_kv(command, "device", cfg.runtime.device)
    _add_kv(command, "dtype", cfg.runtime.dtype)
    _add_kv(command, "compile", cfg.runtime.compile)
    _add_kv(command, "wandb_log", cfg.logging.wandb_log)
    _add_kv(command, "wandb_project", cfg.logging.wandb_project)
    _add_kv(command, "wandb_run_name", cfg.logging.wandb_run_name)

    return command
