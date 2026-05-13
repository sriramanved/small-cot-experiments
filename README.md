# Noisy Imitation Learning Experiments

This repository contains the synthetic noisy imitation learning experiments for
Offline BC, OPD / TM OPD, NAIL-forward, and NAIL-reverse on S5 and modular
addition tasks.

The supported public entrypoint is Hydra:

```sh
python -m nanogpt.run <overrides>
```

Experiment presets live in `hydra_configs/experiment/`, task presets in
`hydra_configs/task/`, and sweep presets in `hydra_configs/sweep/`.

## Install

```sh
pip install torch numpy wandb tqdm hydra-core hydra-submitit-launcher
pip install -e .
```

`wandb` is optional unless `logging=enabled` or `logging.wandb_log=true`.

## Main Pipelines

- `pretrain`: clean teacher training and offline BC on rendered datasets.
- `s5_prompt_bank` / `modadd_prompt_bank`: generate clean prompt banks.
- `s5_render` / `modadd_render`: render offline noisy datasets.
- `opd`: online OPD / TM OPD with student-sampled prefixes.
- `nail`: online NAIL variants with student-collected prefixes.

## Common Runs

Generate an S5 prompt bank:

```sh
python -m nanogpt.run experiment=s5_prompt_bank
```

Train a clean S5 teacher:

```sh
python -m nanogpt.run experiment=s5_cot
```

Render an offline noisy S5 dataset:

```sh
python -m nanogpt.run experiment=s5_noisy_render
```

Train Offline BC on a rendered S5 dataset:

```sh
python -m nanogpt.run experiment=s5_noisy_bc
```

Run OPD / TM OPD:

```sh
python -m nanogpt.run experiment=s5_opd
```

Run NAIL-forward:

```sh
python -m nanogpt.run experiment=s5_nail
```

Run NAIL-reverse with greedy prefix collection and auxiliary reverse-KL actions:

```sh
python -m nanogpt.run experiment=s5_nail_reverse_mc_fixed
```

## Sweeps

Local multirun:

```sh
python -m nanogpt.run --multirun \
  experiment=s5_noisy_bc \
  sweep=s5_noisy_bc_eta \
  task.subset_size=1024
```

Slurm multirun through Hydra Submitit:

```sh
python -m nanogpt.run --multirun \
  experiment=s5_opd \
  sweep=s5_opd_eta \
  hydra/launcher=submitit_slurm \
  cluster=aics \
  cluster.account=<ACCOUNT> \
  cluster.partition=<PARTITION>
```

## Method Controls

Native OPD/NAIL configs use:

- `task.teacher_signal`: `mc` or `full`
- `task.loss`: `forward`, `reverse`, `mixed`, or `jsd`
- `task.kl_beta`: mixture/JSD weight when applicable
- `task.rollout_temperature_override`: prefix collection policy temperature
- `task.loss_temperature_override`: loss-distribution temperature for supported losses

Legacy `task.objective` metadata is still readable for old checkpoints and
analysis scripts, but it is not a supported launch control for native runs.

## Offline Datasets

Offline BC consumes rendered datasets under `data/<dataset_name>/`. The renderer
saves:

- `train_x.pt`, `train_y.pt`
- optional `train_teacher_probs.pt` for full-distribution offline BC
- `val_x.pt`, `val_y.pt`
- `meta.json`

The offline target span is shared with online OPD/NAIL supervision: the full
clean continuation stored in the prompt bank, including the final answer suffix
when present.

## Validation

Useful targeted checks:

```sh
python -m compileall -q src data tests model.py nanogpt_checkpoint.py torch_dtypes.py
python -m unittest tests/test_training_methods.py tests/test_opd_objectives.py
python -m pytest tests/test_hydra_configs.py tests/test_modadd_integration.py tests/test_s5_full_distribution_offline_bc.py -q
python -m pytest -q
```

No lint formatter is currently configured in this repo.
