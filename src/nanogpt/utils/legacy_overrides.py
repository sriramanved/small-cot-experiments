from __future__ import annotations

import sys
from ast import literal_eval


def apply_legacy_overrides(globals_dict: dict[str, object], argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    for arg in argv:
        if "=" not in arg:
            raise ValueError(
                "Python config files are no longer supported. "
                "Use direct --key=value overrides."
            )
        if not arg.startswith("--"):
            raise ValueError(f"Unknown argument format: {arg}")
        key, raw_val = arg.split("=", 1)
        key = key[2:]
        if key not in globals_dict:
            raise ValueError(f"Unknown config key: {key}")
        current = globals_dict[key]
        try:
            value = literal_eval(raw_val)
        except (SyntaxError, ValueError):
            value = raw_val
        if not isinstance(current, type(None)) and type(value) is not type(current):
            raise AssertionError(
                f"Type mismatch for {key}: expected {type(current).__name__}, got {type(value).__name__}"
            )
        print(f"Overriding: {key} = {value}")
        globals_dict[key] = value
