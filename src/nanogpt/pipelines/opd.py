from __future__ import annotations

from nanogpt.config_schema import AppConfig
from nanogpt.utils.repo import script_path


def _add_flag(args: list[str], key: str, enabled: bool) -> None:
    if enabled:
        args.append(f"--{key}")


def _add_kv(args: list[str], key: str, value) -> None:
    if value is None or value == "":
        return
    args.append(f"--{key}={value}")


def build_command(cfg: AppConfig) -> list[str]:
    command = [cfg.runtime.python_bin, str(script_path("train_opd.py"))]
    _add_kv(command, "task", cfg.task.task)
    _add_kv(command, "teacher_checkpoint", cfg.task.teacher_checkpoint)
    _add_kv(command, "prompt_bank_dir", cfg.task.prompt_bank_dir)
    _add_kv(command, "subset_size", cfg.task.subset_size)
    _add_kv(command, "eta", cfg.task.eta)
    _add_kv(command, "teacher_law", cfg.task.teacher_law)
    _add_kv(command, "objective", cfg.task.objective)
    _add_kv(command, "student_temperature", cfg.task.student_temperature)
    _add_kv(command, "out_dir", cfg.run.out_dir)

    for key in (
        "init_from",
        "init_from_ckpt",
        "continue_from_subset_size",
        "batch_size",
        "max_iters",
        "learning_rate",
        "warmup_iters",
        "weight_decay",
        "beta1",
        "beta2",
        "grad_clip",
        "eval_interval",
        "eval_n",
        "eval_batch_size",
        "log_interval",
        "save_interval",
        "seed",
        "eps",
    ):
        _add_kv(command, key, getattr(cfg.optim, key))

    _add_flag(command, "single_epoch", cfg.optim.single_epoch)
    _add_flag(command, "shuffle_prompts", cfg.optim.shuffle_prompts)
    _add_kv(command, "device", cfg.runtime.device)
    _add_kv(command, "dtype", cfg.runtime.dtype)
    _add_flag(command, "compile", cfg.runtime.compile)
    _add_flag(command, "wandb_log", cfg.logging.wandb_log)
    _add_kv(command, "wandb_project", cfg.logging.wandb_project)
    _add_kv(command, "wandb_run_name", cfg.logging.wandb_run_name)
    _add_kv(command, "wandb_run_id", cfg.logging.wandb_run_id)
    _add_kv(command, "wandb_init_timeout", cfg.logging.wandb_init_timeout)
    return command
