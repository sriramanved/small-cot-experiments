from __future__ import annotations

from omegaconf import OmegaConf


def _float_tag(value: object) -> str:
    return str(value).replace(".", "p").replace("-", "neg")


def _temp_tag(value: object) -> str:
    numeric = float(value)
    if numeric == 0:
        return "greedy"
    return f"t{_float_tag(value)}"


def register_resolvers() -> None:
    OmegaConf.register_new_resolver("float_tag", _float_tag, replace=True)
    OmegaConf.register_new_resolver("temp_tag", _temp_tag, replace=True)
