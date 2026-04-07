import os
import torch

_CACHE = {}
_STATE = {}

def _load_split(data_dir, split):
    key = (data_dir, split)
    if key not in _CACHE:
        x = torch.load(os.path.join(data_dir, f"{split}_x.pt"), map_location="cpu")
        y = torch.load(os.path.join(data_dir, f"{split}_y.pt"), map_location="cpu")
        _CACHE[key] = (x, y)
    return _CACHE[key]

def _move(x, y, device):
    x = x.long()
    y = y.long()
    if "cuda" in str(device):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y

def reset_train_epoch(data_dir, shuffle=True, seed=None):
    x_all, _ = _load_split(data_dir, "train")
    n = x_all.size(0)

    if shuffle:
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        perm = torch.randperm(n, generator=g)
    else:
        perm = torch.arange(n)

    _STATE[data_dir] = {
        "perm": perm,
        "pos": 0,
        "n": n,
    }

def get_train_batch_once(batch_size, device, data_dir):
    if data_dir not in _STATE:
        reset_train_epoch(data_dir, shuffle=True)

    x_all, y_all = _load_split(data_dir, "train")
    st = _STATE[data_dir]

    if st["pos"] >= st["n"]:
        raise StopIteration

    idx = st["perm"][st["pos"]: st["pos"] + batch_size]
    st["pos"] += idx.numel()

    return _move(x_all[idx], y_all[idx], device)

def iter_eval_batches(split, batch_size, device, data_dir):
    x_all, y_all = _load_split(data_dir, split)
    n = x_all.size(0)
    for start in range(0, n, batch_size):
        idx = slice(start, min(start + batch_size, n))
        yield _move(x_all[idx], y_all[idx], device)