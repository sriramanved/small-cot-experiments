from __future__ import annotations

from __future__ import annotations

from nanogpt.config_schema import AppConfig
from nanogpt.trainers import run_opd, run_opd_hf, run_pretrain
from nanogpt.trainers.configs import (
    project_opd_config,
    project_opd_hf_config,
    project_pretrain_config,
)


def run_pipeline(cfg: AppConfig, *, launcher_command: list[str]) -> None:
    if cfg.pipeline.name == "pretrain":
        run_pretrain(project_pretrain_config(cfg), launcher_command=launcher_command)
        return
    if cfg.pipeline.name == "opd":
        run_opd(project_opd_config(cfg), launcher_command=launcher_command)
        return
    if cfg.pipeline.name == "opd_hf":
        run_opd_hf(project_opd_hf_config(cfg), launcher_command=launcher_command)
        return
    raise ValueError(f"unsupported pipeline {cfg.pipeline.name!r}")
