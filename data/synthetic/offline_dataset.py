import os

import torch

_CACHE = {}
_STATE = {}


def _effective_n(total_n, subset_size, split):
    if subset_size is None or subset_size <= 0 or split != "train":
        return total_n
    subset_size = int(subset_size)
    if subset_size > total_n:
        raise ValueError(
            f"requested train subset_size={subset_size} exceeds available "
            f"{split} rows={total_n}"
        )
    return subset_size


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


def reset_train_epoch(data_dir, shuffle=True, seed=None, subset_size=None):
    x_all, _ = _load_split(data_dir, "train")
    n = _effective_n(x_all.size(0), subset_size, "train")

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


def get_train_epoch_state(data_dir):
    if data_dir not in _STATE:
        return None
    st = _STATE[data_dir]
    return {
        "perm": st["perm"].clone(),
        "pos": int(st["pos"]),
        "n": int(st["n"]),
    }


def set_train_epoch_state(data_dir, state):
    _STATE[data_dir] = {
        "perm": state["perm"].to(device="cpu", dtype=torch.long).clone(),
        "pos": int(state["pos"]),
        "n": int(state["n"]),
    }


def get_batch(split, batch_size, device, data_dir, subset_size=None):
    x_all, y_all = _load_split(data_dir, split)
    n = _effective_n(x_all.size(0), subset_size, split)
    idx = torch.randint(n, (batch_size,))
    return _move(x_all[idx], y_all[idx], device)


def get_train_batch_once(batch_size, device, data_dir, subset_size=None):
    if data_dir not in _STATE:
        reset_train_epoch(data_dir, shuffle=True, subset_size=subset_size)

    x_all, y_all = _load_split(data_dir, "train")
    st = _STATE[data_dir]

    if st["pos"] >= st["n"]:
        raise StopIteration

    idx = st["perm"][st["pos"]: st["pos"] + batch_size]
    st["pos"] += idx.numel()

    return _move(x_all[idx], y_all[idx], device)


def iter_eval_batches(split, batch_size, device, data_dir, subset_size=None):
    x_all, y_all = _load_split(data_dir, split)
    n = _effective_n(x_all.size(0), subset_size, split)
    for start in range(0, n, batch_size):
        idx = slice(start, min(start + batch_size, n))
        yield _move(x_all[idx], y_all[idx], device)
