from __future__ import annotations

from nanogpt.trainers.configs import NailConfig
from nanogpt.trainers.native_student_prefix import (
    run_student_prefix,
    validate_config,
    validate_resume_metadata as _validate_resume_metadata,
)


def run_nail(cfg: NailConfig, *, launcher_command: list[str]) -> None:
    run_student_prefix(cfg, launcher_command=launcher_command)


validate_args = validate_config


def validate_resume_metadata(out_dir, metadata) -> None:
    _validate_resume_metadata(out_dir, metadata, default_method_family="nail")
