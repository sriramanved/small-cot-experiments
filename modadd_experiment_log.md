# Modular Addition Experiment Log

This log records precise Hydra defaults and concrete run commands for the modadd experiments only.

## Hydra Default Backbone

- Runtime: `device = cuda`, `dtype = float16`, `compile = false`, `backend = nccl`
- Depth-1 model: `n_layer = 1`, `n_head = 8`, `n_embd = 512`, `dropout = 0.0`, `bias = false`
- Shared training defaults: `batch_size = 64`, `learning_rate = 1e-5`, `warmup_iters = 2000`, `weight_decay = 0.0`, `beta1 = 0.9`, `beta2 = 0.95`, `grad_clip = 1.0`
- Derived defaults: `block_size = 2m` for `modadd_cot`, offline BC, and OPD
- OPD default one-pass budget: `max_iters = ceil(subset_size / batch_size)`
- Offline BC default stopping policy: `offline_single_epoch = true`

- `bank_seed`: selects the prompt bank
- `teacher_seed`: selects the expert checkpoint path
- `render_seed`: selects rendered offline dataset identity
- `optim.seed`: training RNG for the current run
- Native online methods now use `task.loss` plus `task.teacher_signal`; do not use the legacy `task.objective` override except for the `opd_hf` pipeline.
- Use `experiment=modadd_nail_reverse_mc_fixed` for fresh NAIL-reverse MC reruns. Its output root includes `nail_reverse_mc_fixed`, so fixed auxiliary-action runs do not get mixed with legacy reverse-NAIL artifacts.

## P=7, M=127 Setup

- Prompt bank: `n_train = 4,000,000`, `n_val = 5,000`, `bank_seed = 1337`
- Expert seed: `20260417`
- Offline subset size: `1,000,000`
- Eta grid: `0.0, 0.1, 0.3, 0.5, 0.7, 0.9`
- Offline BC rollout mode: `sample_then_corrupt`

## Commands

- Prompt bank generation

<details>
<summary>Command</summary>

```bash
./.venv/bin/python -m nanogpt.run \
  experiment=modadd_prompt_bank \
  task.modadd_p=7 \
  task.modadd_m=127 \
  task.n_train=4000000 \
  task.n_val=5000 \
  task.bank_seed=1337
```

</details>

- Clean expert

<details>
<summary>Command</summary>

```bash
./.venv/bin/python -m nanogpt.run \
  experiment=modadd_cot \
  task.modadd_p=7 \
  task.modadd_m=127 \
  task.teacher_seed=20260417 \
  optim.seed=20260417 \
  optim.max_iters=10000 \
  optim.lr_decay_iters=10000 \
  optim.eval_interval=500
```

</details>

- Noisy offline render

<details>
<summary>Command</summary>

```bash
./.venv/bin/python -m nanogpt.run --multirun \
  experiment=modadd_noisy_render \
  task.modadd_p=7 \
  task.modadd_m=127 \
  task.n_train=4000000 \
  task.n_val=5000 \
  task.subset_size=1000000 \
  task.rollout_mode=sample_then_corrupt \
  task.bank_seed=1337 \
  task.teacher_seed=20260417 \
  task.render_seed=20260417 \
  task.eta=0.0,0.1,0.3,0.5,0.7,0.9
```

</details>

- Noisy offline BC MC sweep

<details>
<summary>Command</summary>

```bash
nohup ./.venv/bin/python -m nanogpt.run --multirun \
  experiment=modadd_noisy_bc \
  task.modadd_p=7 \
  task.modadd_m=127 \
  task.bank_seed=1337 \
  task.teacher_seed=20260417 \
  task.render_seed=20260417 \
  task.subset_size=1000000 \
  task.rollout_mode=sample_then_corrupt \
  task.eta=0.0,0.1,0.3,0.5,0.7,0.9 \
  optim.seed=20260417 \
  optim.eval_interval=500 \
  > logs/nohup/modadd_offline_bc_p7_m127_seed20260417.log 2>&1 &
```

</details>

- OPD reverse-KL TM sweep

<details>
<summary>Command</summary>

```bash
nohup ./.venv/bin/python -m nanogpt.run --multirun \
  experiment=modadd_opd \
  task.modadd_p=7 \
  task.modadd_m=127 \
  task.n_train=4000000 \
  task.n_val=5000 \
  task.bank_seed=1337 \
  task.teacher_seed=20260417 \
  task.render_seed=1337 \
  task.subset_size=1000000 \
  task.teacher_signal=mc \
  task.loss=reverse \
  task.rollout_temperature_override=1.0 \
  task.eta=0.0,0.1,0.3,0.5,0.7,0.9 \
  optim.seed=20260417 \
  optim.single_epoch=true \
  optim.eval_interval=500 \
  > logs/nohup/modadd_opd_reverse_kl_tm_p7_m127_seed20260417.log 2>&1 &
```

</details>

- NAIL forward-KL MC sweep

<details>
<summary>Command</summary>

```bash
nohup ./.venv/bin/python -m nanogpt.run --multirun \
  experiment=modadd_nail \
  task.modadd_p=7 \
  task.modadd_m=127 \
  task.n_train=4000000 \
  task.n_val=5000 \
  task.bank_seed=1337 \
  task.teacher_seed=20260417 \
  task.render_seed=1337 \
  task.subset_size=1000000 \
  task.teacher_signal=mc \
  task.loss=forward \
  task.rollout_temperature_override=0.0 \
  task.eta=0.0,0.1,0.3,0.5,0.7,0.9 \
  optim.seed=20260417 \
  optim.single_epoch=true \
  optim.eval_interval=500 \
  > logs/nohup/modadd_nail_forward_mc_p7_m127_seed20260417.log 2>&1 &
```

</details>

- Fixed NAIL reverse-KL MC sweep

<details>
<summary>Command</summary>

```bash
nohup ./.venv/bin/python -m nanogpt.run --multirun \
  experiment=modadd_nail_reverse_mc_fixed \
  task.modadd_p=7 \
  task.modadd_m=127 \
  task.n_train=4000000 \
  task.n_val=5000 \
  task.bank_seed=1337 \
  task.teacher_seed=20260417 \
  task.render_seed=1337 \
  task.subset_size=1000000 \
  task.teacher_signal=mc \
  task.loss=reverse \
  task.rollout_temperature_override=0.0 \
  task.eta=0.0,0.1,0.3,0.5,0.7,0.9 \
  optim.seed=20260417 \
  optim.single_epoch=true \
  optim.eval_interval=500 \
  > logs/nohup/modadd_nail_reverse_mc_fixed_p7_m127_seed20260417.log 2>&1 &
```

</details>

<details>
<summary>Pertinent Notes</summary>

- Offline BC consumes rendered datasets, so `task.render_seed` must match the dataset that exists on disk.
- OPD uses the prompt bank directly; it does not consume the rendered offline BC datasets.
- In Hydra multiruns, per-job logs are written under `hydra_multirun/.../run.log`.

</details>
