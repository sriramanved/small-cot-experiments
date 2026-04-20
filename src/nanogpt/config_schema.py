from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, MISSING, OmegaConf


@dataclass
class PipelineConfig:
    name: str = MISSING
    entrypoint: str = MISSING


@dataclass
class RunConfig:
    output_root: str = "."
    name: str = MISSING
    out_dir: str = MISSING


@dataclass
class LoggingConfig:
    wandb_log: bool = False
    wandb_project: str = "small-cot-experiments"
    wandb_run_name: Optional[str] = None
    wandb_run_id: Optional[str] = None
    wandb_init_timeout: int = 300


@dataclass
class TorchrunConfig:
    nproc_per_node: int = 1
    nnodes: int = 1
    node_rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    standalone: bool = True


@dataclass
class RuntimeConfig:
    python_bin: str = ""
    device: str = "cuda"
    dtype: str = "float16"
    compile: bool = False
    backend: str = "nccl"
    torchrun: TorchrunConfig = field(default_factory=TorchrunConfig)


@dataclass
class ClusterConfig:
    name: str = "local"
    account: Optional[str] = None
    partition: Optional[str] = None
    qos: Optional[str] = None
    gpus_per_node: int = 1
    cpus_per_task: int = 8
    mem_gb: int = 32
    timeout_min: int = 720


@dataclass
class SweepConfig:
    name: str = "none"


@dataclass
class TaskConfig:
    dataset: str = "openwebtext"
    dataset_prefix: str = ""
    run_prefix: str = ""
    out_prefix: str = ""
    data_root: str = "data"
    teacher_output_root: str = "."
    task: str = "s5"
    s5_mode: str = "cot"
    s5_m: int = 21
    modadd_p: int = 7
    modadd_m: int = 21
    bank_seed: int = 1337
    teacher_seed: int = 1337
    render_seed: int = 1337
    teacher_depth: int = 1
    gen_batch_size: int = 1024
    n_train: int = 0
    n_val: int = 0
    subset_size: int = 0
    eta: float = 0.0
    teacher_checkpoint: str = ""
    prompt_bank_dir: str = ""
    teacher_law: str = "distributional_noise"
    objective: str = "reverse_kl_tm"
    student_temperature: float = 1.0
    student_rollout_temperature: float = 0.0
    rollout_mode: str = "greedy_then_corrupt"
    target_mode: str = "tokens"


@dataclass
class ModelConfig:
    n_layer: Optional[int] = None
    n_head: Optional[int] = None
    n_embd: Optional[int] = None
    dropout: Optional[float] = None
    bias: Optional[bool] = None
    block_size: Optional[int] = None


@dataclass
class OptimConfig:
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False
    always_save_checkpoint: bool = True
    init_from: str = "scratch"
    init_from_ckpt: str = ""
    continue_from_subset_size: int = 0
    gradient_accumulation_steps: int = 40
    batch_size: int = 12
    learning_rate: float = 6e-4
    max_iters: int = 600000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5
    save_every: int = 0
    save_interval: int = 0
    single_epoch: bool = False
    shuffle_prompts: bool = False
    seed: int = 1337
    eval_n: int = 5000
    eval_batch_size: int = 256
    eps: float = 1e-10
    offline_single_epoch: bool = False
    offline_eval_full: bool = True
    offline_train_subset_size: int = 0
    offline_train_shuffle: bool = False
    offline_target_type: str = "tokens"
    final_eval_on_exit: bool = False
    s5_eval_metrics: bool = False
    s5_eval_clean_train_loss: bool = False
    modadd_eval_metrics: bool = False
    modadd_eval_clean_train_loss: bool = False
    s5_eval_n: int = 256
    s5_eval_batch_size: int = 256
    s5_eval_seed: int = 123


@dataclass
class AppConfig:
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    run: RunConfig = field(default_factory=RunConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)


def materialize_config(raw_cfg: DictConfig) -> AppConfig:
    structured = OmegaConf.structured(AppConfig)
    merged = OmegaConf.merge(structured, raw_cfg)
    cfg = OmegaConf.to_object(merged)
    assert isinstance(cfg, AppConfig)

    if not cfg.runtime.python_bin:
        cfg.runtime.python_bin = sys.executable
    if not cfg.logging.wandb_run_name:
        cfg.logging.wandb_run_name = cfg.run.name
    if not cfg.run.out_dir:
        cfg.run.out_dir = str(Path(cfg.run.output_root) / cfg.run.name)

    if cfg.pipeline.name not in {
        "pretrain",
        "opd",
        "opd_hf",
        "modadd_prompt_bank",
        "modadd_render",
        "s5_prompt_bank",
        "s5_render",
    }:
        raise ValueError(f"unsupported pipeline {cfg.pipeline.name!r}")
    if cfg.pipeline.name == "pretrain" and not cfg.task.dataset:
        raise ValueError("pretrain experiments require task.dataset")
    if cfg.pipeline.name in {"opd", "opd_hf"}:
        if not cfg.task.teacher_checkpoint:
            raise ValueError(f"{cfg.pipeline.name} experiments require task.teacher_checkpoint")
        if not cfg.task.prompt_bank_dir:
            raise ValueError(f"{cfg.pipeline.name} experiments require task.prompt_bank_dir")
        if cfg.task.subset_size <= 0:
            raise ValueError(f"{cfg.pipeline.name} experiments require task.subset_size > 0")
    if cfg.pipeline.name == "modadd_prompt_bank":
        if not cfg.task.prompt_bank_dir:
            raise ValueError("modadd_prompt_bank experiments require task.prompt_bank_dir")
        if cfg.task.n_train <= 0 or cfg.task.n_val <= 0:
            raise ValueError("modadd_prompt_bank experiments require task.n_train > 0 and task.n_val > 0")
    if cfg.pipeline.name == "modadd_render":
        if not cfg.task.teacher_checkpoint:
            raise ValueError("modadd_render experiments require task.teacher_checkpoint")
        if not cfg.task.prompt_bank_dir:
            raise ValueError("modadd_render experiments require task.prompt_bank_dir")
        if not cfg.task.dataset:
            raise ValueError("modadd_render experiments require task.dataset")
        if cfg.task.subset_size <= 0:
            raise ValueError("modadd_render experiments require task.subset_size > 0")
    if cfg.pipeline.name == "s5_prompt_bank":
        if not cfg.task.prompt_bank_dir:
            raise ValueError("s5_prompt_bank experiments require task.prompt_bank_dir")
        if cfg.task.n_train <= 0 or cfg.task.n_val <= 0:
            raise ValueError("s5_prompt_bank experiments require task.n_train > 0 and task.n_val > 0")
    if cfg.pipeline.name == "s5_render":
        if not cfg.task.teacher_checkpoint:
            raise ValueError("s5_render experiments require task.teacher_checkpoint")
        if not cfg.task.prompt_bank_dir:
            raise ValueError("s5_render experiments require task.prompt_bank_dir")
        if not cfg.task.dataset:
            raise ValueError("s5_render experiments require task.dataset")
        if cfg.task.subset_size <= 0:
            raise ValueError("s5_render experiments require task.subset_size > 0")
    return cfg
