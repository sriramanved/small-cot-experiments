from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def script_path(script_name: str) -> Path:
    return repo_root() / script_name


def command_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(repo_root() / "src")
    repo_path = str(repo_root())
    existing = env.get("PYTHONPATH", "")
    pieces = [src_path, repo_path]
    if existing:
        pieces.append(existing)
    env["PYTHONPATH"] = ":".join(pieces)
    return env


def write_launch_metadata(out_dir: Path, *, cfg: Any, command: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(cfg) if is_dataclass(cfg) else cfg
    with open(out_dir / "launcher_config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(out_dir / "launcher_command.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(command) + "\n")


def run_command(command: list[str]) -> None:
    subprocess.run(
        command,
        cwd=repo_root(),
        env=command_env(),
        check=True,
    )
