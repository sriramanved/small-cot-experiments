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