from __future__ import annotations

from nanogpt.config_schema import AppConfig
from nanogpt.pipelines import opd, opd_hf, pretrain


def build_command(cfg: AppConfig) -> list[str]:
    if cfg.pipeline.name == "pretrain":
        return pretrain.build_command(cfg)
    if cfg.pipeline.name == "opd":
        return opd.build_command(cfg)
    if cfg.pipeline.name == "opd_hf":
        return opd_hf.build_command(cfg)
    raise ValueError(f"unsupported pipeline {cfg.pipeline.name!r}")
