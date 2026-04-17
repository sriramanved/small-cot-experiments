"""
Legacy key/value override parser used by the trainer backends.

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

Python config files are no longer supported; use the Hydra entrypoint instead:
`python -m nanogpt.run experiment=<preset> ...`
"""

import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if '=' not in arg:
        raise ValueError(
            "Python config files are no longer supported. "
            "Use `python -m nanogpt.run experiment=<preset> ...`."
        )
    else:
        # assume it's a --key=value argument
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                # attempt to eval it it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            # ensure the types match ok
            assert type(attempt) == type(globals()[key])
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
