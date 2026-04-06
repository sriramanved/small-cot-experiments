import os
import torch

_CACHE = {}

def _load_split(data_dir, split):
    key = (data_dir, split)
    if key not in _CACHE:
        x = torch.load(os.path.join(data_dir, f"{split}_x.pt"), map_location="cpu")
        y = torch.load(os.path.join(data_dir, f"{split}_y.pt"), map_location="cpu")
        _CACHE[key] = (x, y)
    return _CACHE[key]

def get_batch(split, batch_size, device, data_dir):
    x_all, y_all = _load_split(data_dir, split)
    ix = torch.randint(x_all.size(0), (batch_size,))
    x = x_all[ix].long()
    y = y_all[ix].long()

    if "cuda" in str(device):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y