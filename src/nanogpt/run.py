from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from nanogpt.config_schema import materialize_config
from nanogpt.pipelines import build_command
from nanogpt.utils.repo import run_command, write_launch_metadata
from nanogpt.utils.resolvers import register_resolvers


register_resolvers()


@hydra.main(version_base=None, config_path="../../hydra_configs", config_name="config")
def main(raw_cfg: DictConfig) -> None:
    cfg = materialize_config(raw_cfg)
    command = build_command(cfg)
    write_launch_metadata(Path(cfg.run.out_dir), cfg=cfg, command=command)
    print(OmegaConf.to_yaml(raw_cfg))
    print("launch:", " ".join(command))
    run_command(command)


if __name__ == "__main__":
    main()
