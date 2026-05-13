# Noisy Imitation Learning Experiments

This repository contains the synthetic noisy imitation learning experiments for
LogLossBC, NAIL-F, NAIL-R, OPD-F, and OPD-R on S5 and modular-addition tasks.

Readers coming from the paper should start with
[`experiment_log.md`](experiment_log.md) for commands and reproducibility, then
read [`docs/methods.md`](docs/methods.md) for the method taxonomy and
implementation semantics.

## Reader's Guide

The core implementation separates four choices that are bundled together in the
paper notation: how learner prefixes are generated, which noisy teacher law is
queried, which token/distribution supervises the next step, and which KL
direction or surrogate is optimized.

| What to inspect | Primary files |
|---|---|
| Hydra entrypoint and config schema | `src/nanogpt/run.py`, `src/nanogpt/config_schema.py` |
| Pipeline dispatch | `src/nanogpt/pipelines/__init__.py` |
| LogLossBC / clean teacher training | `src/nanogpt/workers/pretrain_body.py`, `data/synthetic/offline_dataset.py` |
| Student-prefix methods | `src/nanogpt/trainers/native_student_prefix.py`, `src/nanogpt/methods/student_prefix.py` |
| S5 task and render path | `data/s5_cot/task.py`, `data/s5_cot/offline_render.py` |
| Modular-addition task and render path | `data/modular_addition/task.py`, `data/modular_addition/offline_render.py` |
| Absorbing random-suffix law | `data/synthetic/random_suffix_noise.py` |

## Implementation Backend Vs Paper Method

`student_prefix` is the shared implementation backend for NAIL-F, NAIL-R,
OPD-F, and OPD-R. Use `pipeline=student_prefix` for new student-prefix configs.
Historical Hydra pipeline names still work: `pipeline=nail` is the old
greedy-default name, while `pipeline=opd` is the sampled-default OPD-R
entrypoint.
Paper method names are presets over rollout temperature, loss direction, and
teacher signal.

| Paper method | Backend/trainer | Prefix policy | Loss sample | Teacher signal | Canonical launch |
|---|---|---|---|---|---|
| LogLossBC | `pretrain` / `src/nanogpt/workers/pretrain_body.py` | Fixed noisy expert rollouts | Rendered token, or saved teacher distribution | None online | `experiment=s5_noisy_bc` / `experiment=modadd_noisy_bc` |
| NAIL-F | `student_prefix` / `src/nanogpt/trainers/native_student_prefix.py` | Greedy student prefixes | Teacher-sampled token for MC forward loss | `mc`, or `full` for full KL | `experiment=s5_nail` / `experiment=modadd_nail` |
| NAIL-R | `student_prefix` / `src/nanogpt/trainers/native_student_prefix.py` | Greedy student prefixes | Fresh auxiliary student token | `mc` | `experiment=s5_nail_reverse_mc_fixed` / `experiment=modadd_nail_reverse_mc_fixed` |
| OPD-F | `student_prefix` / `src/nanogpt/trainers/native_student_prefix.py` | Sampled student prefixes | Teacher-sampled token for MC forward loss | `mc`, or `full` for full KL | `experiment=s5_opd_forward` / `experiment=modadd_opd_forward` |
| OPD-R | `student_prefix` / `src/nanogpt/trainers/native_student_prefix.py` | Sampled student prefixes | Rollout token reused as reverse sample | `mc` | `experiment=s5_opd` / `experiment=modadd_opd` |

OPD-F shares the student-prefix backend with NAIL-F/R, but it is conceptually
OPD because it uses sampled student prefixes rather than greedy prefixes.
OPD-R uses the `opd` pipeline entrypoint, but that entrypoint delegates to the
shared `run_student_prefix` backend; there is not a second OPD-R
implementation.

| Paper notation / concept | Code name |
|---|---|
| Student policy | `student`, current model logits in `run_student_prefix` |
| Greedy student rollout | `task.rollout_temperature_override=0.0` |
| Sampled student rollout | `task.rollout_temperature_override=1.0` or default `opd` rollout |
| Rollout temperature | `task.rollout_temperature_override`; affects prefix collection only |
| Loss distribution temperature | `task.loss_temperature_override`; optional loss-side distribution temperature |
| Clean expert | `task.teacher_checkpoint` |
| Noisy expert | `task.teacher_law` plus `task.eta` |
| Corruption rate eta | `task.eta` |
| KL mixture beta | `task.kl_beta`; `0` is forward-heavy, `1` is reverse-heavy |
| Teacher token | `teacher_targets` sampled by `sample_teacher_actions` |
| Auxiliary student token | `aux_actions` from `sample_student_aux_actions` |

Scope note: this repo covers the synthetic S5 and modular-addition experiments.
GSM8K code is maintained separately.

Some implementation objectives are empirical stopped-prefix surrogates of the
literal augmented-law objectives in the theory. In particular, rollout samples
are generated under `torch.no_grad()`, and temperature-mismatched sampled
rollouts should be read as a surrogate unless the rollout distribution matches
the temperature-one student distribution used in the loss.

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

- `pretrain`: clean teacher training and LogLossBC on rendered datasets.
- `s5_prompt_bank` / `modadd_prompt_bank`: generate clean prompt banks.
- `s5_render` / `modadd_render`: render offline noisy datasets.
- `student_prefix`: neutral online backend for NAIL-F/R and OPD-F aliases.
- `opd`: OPD-R entrypoint; it delegates to the shared `run_student_prefix`
  backend with sampled rollout defaults.
- `nail`: legacy greedy-default alias for the student-prefix backend.

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

Train LogLossBC on a rendered S5 dataset:

```sh
python -m nanogpt.run experiment=s5_noisy_bc
```

Run OPD-R:

```sh
python -m nanogpt.run experiment=s5_opd
```

Run NAIL-F:

```sh
python -m nanogpt.run experiment=s5_nail
```

Run OPD-F:

```sh
python -m nanogpt.run experiment=s5_opd_forward
```

Modular-addition OPD-F:

```sh
python -m nanogpt.run experiment=modadd_opd_forward
```

Run NAIL-R with greedy prefix collection and auxiliary reverse-KL actions:

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

Student-prefix method configs use:

- `task.teacher_signal`: `mc` or `full`
- `task.loss`: `forward`, `reverse`, `mixed`, or `jsd`
- `task.kl_beta`: mixture/JSD weight when applicable
- `task.rollout_temperature_override`: prefix collection policy temperature
- `task.loss_temperature_override`: loss-distribution temperature for supported losses

Legacy `task.objective` metadata is still readable for old checkpoints and
analysis scripts, but it is not a supported launch control for native runs.
Old objective strings such as `forward_kl_simple`, `forward_kl_full`,
`reverse_kl_simple`, `reverse_kl_tm`, and `reverse_kl_full` normalize to the
canonical fields at load time and should not be used for new runs.

## Offline Datasets

LogLossBC consumes rendered datasets under `data/<dataset_name>/`. The renderer
saves:

- `train_x.pt`, `train_y.pt`
- optional `train_teacher_probs.pt` for full-distribution LogLossBC
- `val_x.pt`, `val_y.pt`
- `meta.json`

The offline target span is shared with the student-prefix objectives: the full
clean continuation stored in the prompt bank, including the final answer suffix
when present.

## Reproducibility Notes

Minimal sanity checks are intentionally small and CPU-friendly:

```sh
python -m compileall -q src data scripts tests model.py nanogpt_checkpoint.py torch_dtypes.py
python -m unittest tests/test_training_methods.py tests/test_opd_objectives.py
python -m pytest tests/test_hydra_configs.py tests/test_modadd_integration.py tests/test_s5_full_distribution_offline_bc.py -q
```

Paper-scale runs should be launched from the command templates in
[`experiment_log.md`](experiment_log.md). The relevant config families are
`hydra_configs/experiment/s5_*`, `hydra_configs/experiment/modadd_*`, and the
matching task files under `hydra_configs/task/`.

Expected qualitative outcomes for the main synthetic checks:

- Clean `eta=0`: LogLossBC should learn fastest, consistent with clean expert
  supervision being optimal in the offline setting.
- Absorbing modular-addition noise at `eta=0.2`: NAIL-F/NAIL-R should solve the
  task, while LogLossBC and sampled-rollout OPD baselines are expected to
  struggle because most fixed noisy trajectories contain uninformative suffixes.

No lint formatter is currently configured in this repo.
