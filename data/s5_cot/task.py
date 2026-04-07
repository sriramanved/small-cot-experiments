import random
import torch

TOKENS = ['(', ')', '='] + [str(i) for i in range(1, 6)]
stoi = {t: i for i, t in enumerate(TOKENS)}
itos = {i: t for t, i in stoi.items()}
VOCAB_SIZE = len(TOKENS)  # 8

def encode(tokens):
    return [stoi[t] for t in tokens]

def decode(ids):
    return [itos[int(i)] for i in ids]

def compose(sigma, pi):
    # paper: sigma o pi = (sigma_{pi_1}, ..., sigma_{pi_5})
    return [sigma[j - 1] for j in pi]

def perm_tokens(perm):
    return ['('] + [str(x) for x in perm] + [')']

def sample_perm():
    p = [1, 2, 3, 4, 5]
    random.shuffle(p)
    return p

def sample_cot_example(m=21):
    prompt = []
    cot = []
    running = None

    for _ in range(m):
        sigma = sample_perm()
        prompt.extend(perm_tokens(sigma))
        running = sigma if running is None else compose(running, sigma)
        cot.extend(perm_tokens(running))

    prompt.append('=')
    return prompt, cot

def sample_base_example(m=21):
    prompt = []
    running = None

    for _ in range(m):
        sigma = sample_perm()
        prompt.extend(perm_tokens(sigma))
        running = sigma if running is None else compose(running, sigma)

    prompt.append('=')
    target = perm_tokens(running)
    return prompt, target

def sample_xy(mode='cot', m=21):
    if mode == 'cot':
        prompt, target = sample_cot_example(m=m)
    elif mode == 'base':
        prompt, target = sample_base_example(m=m)
    else:
        raise ValueError(mode)

    seq = prompt + target
    ids = encode(seq)

    x = torch.tensor(ids[:-1], dtype=torch.long)
    y = torch.tensor(ids[1:], dtype=torch.long)

    prompt_len = len(prompt)
    y[:prompt_len - 1] = -1  # only compute loss on continuation
    return x, y

def get_batch(batch_size, device, mode='cot', m=21):
    xs, ys = [], []
    for _ in range(batch_size):
        x, y = sample_xy(mode=mode, m=m)
        xs.append(x)
        ys.append(y)
    X = torch.stack(xs).to(device)
    Y = torch.stack(ys).to(device)
    return X, Y

def continuation_exact_from_logits(logits, y):
    pred = logits.argmax(dim=-1)
    mask = (y != -1)
    ok = torch.logical_or(pred.eq(y), ~mask)
    return ok.all(dim=1).float().mean().item()

# for training the noisy expert

def sample_prompt_only(m=21):
    prompt = []
    for _ in range(m):
        sigma = sample_perm()
        prompt.extend(perm_tokens(sigma))
    prompt.append('=')
    return prompt

CORRUPTIBLE_IDS = [stoi[str(i)] for i in range(1, 6)]  # for S5

def corrupt_token(tok_id, eta):
    if tok_id in CORRUPTIBLE_IDS and random.random() < eta:
        return random.choice(CORRUPTIBLE_IDS)
    return tok_id

def sample_perm_from_rng(rng):
    p = [1, 2, 3, 4, 5]
    rng.shuffle(p)
    return p

def sample_cot_example_from_rng(rng, m=21):
    prompt = []
    cot = []
    running = None
    for _ in range(m):
        sigma = sample_perm_from_rng(rng)
        prompt.extend(perm_tokens(sigma))
        running = sigma if running is None else compose(running, sigma)
        cot.extend(perm_tokens(running))
    prompt.append('=')
    return prompt, cot

@torch.no_grad()
def greedy_generate_ids(model, prompt_ids, max_new_tokens, device):
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)
    return idx[0].tolist()

@torch.no_grad()
def evaluate_clean_s5_metrics(
    model,
    device,
    n_eval=256,
    m=21,
    seed=123,
):
    """
    Returns three clean-task metrics:
      - cot_exact: teacher-forced full-CoT exact
      - clean_full_exact: autoregressive full-CoT exact
      - clean_final_exact: autoregressive final-answer exact
    """
    rng = random.Random(seed)

    tf_full_ok = 0
    ar_full_ok = 0
    ar_final_ok = 0

    for _ in range(n_eval):
        prompt_toks, cot_toks = sample_cot_example_from_rng(rng, m=m)
        prompt_ids = encode(prompt_toks)
        cot_ids = encode(cot_toks)

        # teacher-forced full-CoT exact
        seq = prompt_ids + cot_ids
        x = torch.tensor([seq[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([seq[1:]], dtype=torch.long, device=device)
        y[:, :len(prompt_ids)-1] = -1

        logits, _ = model(x, y)
        pred = logits.argmax(dim=-1)
        mask = (y != -1)
        tf_ok = torch.logical_or(pred.eq(y), ~mask).all(dim=1).item()
        tf_full_ok += int(tf_ok)

        # autoregressive full / final exact
        out_ids = greedy_generate_ids(model, prompt_ids, len(cot_ids), device)
        pred_ids = out_ids[len(prompt_ids):]

        if pred_ids == cot_ids:
            ar_full_ok += 1
        if pred_ids[-7:] == cot_ids[-7:]:
            ar_final_ok += 1

    return {
        "cot_exact": tf_full_ok / n_eval,
        "clean_full_exact": ar_full_ok / n_eval,
        "clean_final_exact": ar_final_ok / n_eval,
    }

import os

@torch.no_grad()
def evaluate_saved_clean_s5_metrics(
    model,
    device,
    data_dir,
    n_eval=None,
):
    prompt_ids_all = torch.load(os.path.join(data_dir, "clean_val_prompt_ids.pt"), map_location="cpu").long()
    cot_ids_all = torch.load(os.path.join(data_dir, "clean_val_cot_ids.pt"), map_location="cpu").long()

    if n_eval is not None:
        prompt_ids_all = prompt_ids_all[:n_eval]
        cot_ids_all = cot_ids_all[:n_eval]

    tf_full_ok = 0
    ar_full_ok = 0
    ar_final_ok = 0
    n = prompt_ids_all.size(0)

    for i in range(n):
        prompt_ids = prompt_ids_all[i].tolist()
        cot_ids = cot_ids_all[i].tolist()

        # teacher-forced full-CoT exact
        seq = prompt_ids + cot_ids
        x = torch.tensor([seq[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([seq[1:]], dtype=torch.long, device=device)
        y[:, :len(prompt_ids)-1] = -1

        logits, _ = model(x, y)
        pred = logits.argmax(dim=-1)
        mask = (y != -1)
        tf_ok = torch.logical_or(pred.eq(y), ~mask).all(dim=1).item()
        tf_full_ok += int(tf_ok)

        # autoregressive full / final exact
        out_ids = greedy_generate_ids(model, prompt_ids, len(cot_ids), device)
        pred_ids = out_ids[len(prompt_ids):]

        if pred_ids == cot_ids:
            ar_full_ok += 1
        if pred_ids[-7:] == cot_ids[-7:]:
            ar_final_ok += 1

    return {
        "cot_exact": tf_full_ok / n,
        "clean_full_exact": ar_full_ok / n,
        "clean_final_exact": ar_final_ok / n,
    }