# Method Guide

This repo exposes the paper's online-method decomposition as explicit config
knobs:

- Prefix-generation policy / rollout distribution:
  `task.rollout_temperature_override`
- Local per-prefix loss or KL direction:
  `task.loss=forward|reverse|mixed|jsd`
- Teacher signal type:
  `task.teacher_signal=mc|full`
- Implementation backend:
  `student_prefix`, implemented by `src/nanogpt/trainers/native_student_prefix.py`

`student_prefix` is the shared backend for NAIL-F, NAIL-R, OPD-F, and OPD-R.
Use `pipeline=student_prefix` for new student-prefix configs. Historical Hydra
pipeline names remain compatibility entrypoints: `pipeline=nail` is the old
greedy-default name. OPD-R uses the `opd` pipeline entrypoint, but that
entrypoint delegates to the shared `run_student_prefix` backend with sampled
rollout defaults.

| Paper method | Backend/trainer | Prefix policy | Loss sample | Teacher signal | Canonical launch |
|---|---|---|---|---|---|
| LogLossBC | `pretrain` / `src/nanogpt/workers/pretrain_body.py` | Fixed noisy expert rollouts | Rendered dataset token, or saved teacher distribution | None online | `experiment=s5_noisy_bc` / `experiment=modadd_noisy_bc` |
| NAIL-F | `student_prefix` / `native_student_prefix.py` | Greedy student prefixes | Teacher-sampled token for MC forward loss | `mc`, or `full` for full KL | `experiment=s5_nail` / `experiment=modadd_nail` |
| NAIL-R | `student_prefix` / `native_student_prefix.py` | Greedy student prefixes | Fresh auxiliary student token | `mc` | `experiment=s5_nail_reverse_mc_fixed` / `experiment=modadd_nail_reverse_mc_fixed` |
| OPD-F | `student_prefix` / `native_student_prefix.py` | Sampled student prefixes | Teacher-sampled token for MC forward loss | `mc`, or `full` for full KL | `experiment=s5_opd_forward` / `experiment=modadd_opd_forward` |
| OPD-R | `student_prefix` / `native_student_prefix.py` | Sampled student prefixes | Rollout token reused as reverse sample | `mc` | `experiment=s5_opd` / `experiment=modadd_opd` |

OPD-F shares the student-prefix backend with NAIL-F/R, but it is conceptually
OPD because it uses sampled student prefixes rather than greedy prefixes.
OPD-R also uses the same backend: `pipeline=opd` is only the entrypoint name,
and it delegates to `run_student_prefix`.

## Paper Object To Code

| Paper object | Code object/function | Notes |
|---|---|---|
| `\bar\pi_\theta` greedy rollout policy | `rollout_student(..., temperature=0.0)` in `src/nanogpt/methods/student_prefix.py` | Prefix collection policy for NAIL-F/R. |
| `\pi_\theta(\cdot | prefix)` loss-side student distribution | `p_answer_logits` in `run_student_prefix` | Gradients flow through this distribution at fixed visited prefixes. |
| Noisy teacher `\pi^\star_\eta` | `cached_teacher_token_probs` plus `compute_teacher_token_probs` | Clean teacher logits are queried on learner prefixes, then the noisy law is applied. |
| Forward MC loss | `forward_kl_simple_loss` / `forward_mc_loss` | Uses a token sampled from the noisy teacher distribution. |
| Full forward KL | `forward_kl_full_loss` / `forward_full_kl_loss` | Uses the full noisy teacher distribution. |
| Reverse MC estimator | `reverse_kl_tm_loss` / `reverse_kl_mc_loss` | Score-function estimator with gradients only through current loss-side student logits. |
| Auxiliary token | `sample_student_aux_actions` | NAIL-R draws this separately from greedy rollout tokens. |
| Rollout token reused for OPD-R | `select_reverse_mc_actions(method_family="opd", ...)` | Reuses sampled rollout actions when the rollout distribution matches the loss distribution. |
| Rollout temperature | `task.rollout_temperature_override` | Controls prefix collection only. |
| Loss temperature | `task.loss_temperature_override` | Controls the loss-side student distribution where supported; it does not change prefix collection. |
| Teacher signal type | `task.teacher_signal=mc|full` | Selects sampled teacher token versus full teacher distribution. |
| Local loss direction | `task.loss=forward|reverse|mixed|jsd` | Selects the per-prefix objective. |

## Code Walkthrough By Training Step

Student-prefix online methods all pass through `run_student_prefix` in
`src/nanogpt/trainers/native_student_prefix.py`.

| Step | Function / source | Invariant |
|---|---|---|
| Config normalization / legacy aliases | `normalize_student_prefix_method`, `src/nanogpt/methods/student_prefix.py` | Legacy objective strings normalize to canonical `loss`, `teacher_signal`, and resolved temperatures before metadata is written. |
| Prompt batch selection | `FixedPromptCycle.next_batch_indices`, `src/nanogpt/methods/student_prefix.py` | All online methods train on the selected prompt-bank subset without changing the prompt order semantics. |
| Student rollout / prefix collection | `rollout_student`, `src/nanogpt/methods/student_prefix.py` | Rollout temperature controls only the visited prefixes; rollout is under `torch.no_grad()`. |
| Clean teacher logits | `cached_teacher_token_probs`, `src/nanogpt/methods/student_prefix.py` | The frozen clean teacher is queried on learner-induced prefixes. |
| Noisy teacher law application | `compute_teacher_token_probs`, `semantic_key_noise_probs`, random-suffix helpers in `src/nanogpt/methods/student_prefix.py` | `task.teacher_law` and `task.eta` transform clean teacher logits into noisy next-token distributions. |
| NAIL-F / OPD-F action selection | `sample_teacher_actions` or full `teacher_probs` in `run_student_prefix` | MC forward uses a teacher-sampled token; full forward uses the full noisy teacher distribution. |
| NAIL-R action selection | `sample_student_aux_actions` via `select_reverse_mc_actions` | NAIL-R samples a fresh auxiliary student token at the fixed greedy prefix. |
| OPD-R action selection | `select_reverse_mc_actions(method_family="opd", ...)` | OPD-R reuses the sampled rollout token when rollout and loss distributions match; mismatched temperatures are surrogates. |
| Loss computation | `forward_kl_simple_loss`, `forward_kl_full_loss`, `reverse_kl_tm_loss`, `reverse_kl_full_loss`, `mixed_kl_loss_from_components`, `jsd_mc_loss` | Loss gradients update the current next-token distribution at fixed prefixes, not the rollout sampling path. |
| Metadata/logging | `build_run_metadata`, `save_run_metadata`, `maybe_init_wandb` | `run_meta.json` includes `resolved_method_name`, rollout policy/temperature, `loss`, `teacher_signal`, and compatibility fields when present. |

## LogLossBC

Paper notation:
LogLossBC trains on fixed trajectories from the noisy expert law `pi_eta`, after
those trajectories have already been rendered into an offline dataset.

Semantics:
The trainer does not query the teacher online. Downstream corrupted prefixes are
part of the dataset because rendering feeds realized noisy tokens back into the
teacher trajectory.

Config knobs:
`pipeline=pretrain`, `task.dataset`, `task.teacher_law`, `task.eta`,
`task.rollout_mode`, `task.target_mode=tokens|teacher_probs`.

Source pointers:
`src/nanogpt/workers/pretrain_body.py`, `data/synthetic/offline_dataset.py`,
`data/synthetic/offline_losses.py`, and the task-specific renderers under
`data/s5_cot/offline_render.py` and `data/modular_addition/offline_render.py`.

Minimal launch:

```bash
python -m nanogpt.run experiment=s5_noisy_bc
```

## NAIL-F

Paper notation:
Prefixes are collected from the greedy learner policy. On each visited prefix,
the update is the forward local objective, either MC cross-entropy from a teacher
sample or exact `KL(pi_eta || pi_theta)` when the full teacher distribution is
requested.

Semantics:
`task.rollout_temperature_override=0.0` controls prefix collection only.
`task.loss_temperature_override`, when set, controls the loss-side student
distribution for the forward loss and does not change which prefixes are
visited.

Config knobs:
`task.loss=forward`, `task.teacher_signal=mc|full`,
`task.rollout_temperature_override=0.0`, optional
`task.loss_temperature_override`.

Source pointers:
`run_student_prefix` in `src/nanogpt/trainers/native_student_prefix.py`,
`rollout_student`, `forward_mc_loss`, and `forward_full_kl_loss` in
`src/nanogpt/methods/student_prefix.py`.

Minimal launch:

```bash
python -m nanogpt.run experiment=s5_nail
```

## NAIL-R

Paper notation:
Prefixes are collected from the greedy learner policy. The reverse local
objective is estimated with a separate auxiliary sample from the student
distribution at the fixed visited prefix.

Semantics:
The auxiliary student token is distinct from the greedy rollout token. The
greedy token builds the prefix; the auxiliary token drives the reverse-KL
score-function estimator.

Config knobs:
`task.loss=reverse`, `task.teacher_signal=mc`,
`task.rollout_temperature_override=0.0`. Reverse MC does not support
`task.loss_temperature_override`.

Source pointers:
`select_reverse_mc_actions` and `run_student_prefix` in
`src/nanogpt/trainers/native_student_prefix.py`, plus
`sample_student_aux_actions` and `reverse_kl_mc_loss` in
`src/nanogpt/methods/student_prefix.py`.

Minimal launch:

```bash
python -m nanogpt.run experiment=s5_nail_reverse_mc_fixed
```

## OPD-F

Paper notation:
Prefixes are collected from sampled learner rollouts. On each sampled prefix,
the update is the forward local objective, either MC teacher-token
cross-entropy or exact forward KL when `teacher_signal=full`.

Semantics:
This is not conceptually a NAIL method. It shares the `student_prefix` backend
with NAIL-F/R, but `task.loss=forward` with sampled rollout is OPD-F.

Config knobs:
`task.loss=forward`, `task.teacher_signal=mc`, and
`task.rollout_temperature_override=1.0`. To run the full-distribution variant,
override `task.teacher_signal=full`.

Source pointers:
`hydra_configs/experiment/s5_opd_forward.yaml`,
`hydra_configs/experiment/modadd_opd_forward.yaml`,
`run_student_prefix` in `src/nanogpt/trainers/native_student_prefix.py`, and
`forward_mc_loss` / `forward_full_kl_loss` in
`src/nanogpt/methods/student_prefix.py`.

Minimal launch:

```bash
python -m nanogpt.run experiment=s5_opd_forward
```

## OPD-R

Paper notation:
Prefixes are collected from sampled learner rollouts, and the reverse local
objective uses the sampled student action on that same prefix.

Semantics:
The rollout token is reused as the reverse-KL sample when the rollout
temperature matches the loss distribution. Temperature-mismatched sampled
rollouts are stopped-prefix surrogates rather than the literal temperature-one
reverse-KL estimator.

Config knobs:
`task.loss=reverse`, `task.teacher_signal=mc`, and sampled rollout by default
through `experiment=s5_opd` or `experiment=modadd_opd`. Use
`task.rollout_temperature_override` only for explicit rollout-temperature
ablations.

Source pointers:
`select_reverse_mc_actions` and `run_student_prefix` in
`src/nanogpt/trainers/native_student_prefix.py`, plus `reverse_kl_mc_loss` in
`src/nanogpt/methods/student_prefix.py`.

Minimal launch:

```bash
python -m nanogpt.run experiment=s5_opd
```

## Compatibility Note

Old checkpoints/configs may contain legacy objective strings such as
`forward_kl_simple`, `forward_kl_full`, `reverse_kl_simple`, `reverse_kl_tm`,
and `reverse_kl_full`, or legacy temperature fields such as
`student_temperature` and `student_rollout_temperature`. These are normalized at
load time and should not be used for new runs.
