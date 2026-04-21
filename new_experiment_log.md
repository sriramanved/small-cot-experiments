# New S5 Experiment Log

This log tracks the current S5, `m = 21` experiment suite under the native Hydra setup.

## Overview

For the S5, `m = 21` task, we first trained a clean expert with online CoT training for `100k` iterations.

Expert run details:

- Experiment: `s5_cot_len21`
- Teacher seed: `20260417`
- Optim seed: `20260417`
- Intended training budget: `optim.max_iters=100000`, `optim.lr_decay_iters=100000`
- Output directory: `reruns/s5_m21_teacher20260417/out-s5-cot-m21-depth1-seed20260417`
- Fill in final metrics from `last_eval.json`: `[insert final val/loss, clean_full_exact, clean_final_exact]`
- Fill in any provenance notes: `[insert notes]`

We then use the fixed prompt bank
`data/s5_clean_prompt_bank_m21_n15000000_val5000`.

This prompt bank stores:

- `clean_train_prompt_ids.pt`
- `clean_train_cot_ids.pt`
- `clean_val_prompt_ids.pt`
- `clean_val_cot_ids.pt`
- `train_order.pt`
- `meta.json`

The consistency policy for the current S5 comparisons is:

- keep `bank_seed = 1337` fixed across all compared runs
- use the same prompt bank directory across methods
- use the same validation split copied from the prompt bank
- use fixed ordered prefix subsets of the same `train_order.pt`
- keep the prompt order unshuffled during training
- vary `teacher_seed`, `optim.seed`, and `render_seed` only when intentionally changing experiment seed families

This gives us fixed-order strict subsets of one common prompt bank. In particular:

- the `8M` subset is the prefix `train_order[:8000000]`
- the `12M` subset is the prefix `train_order[:12000000]`
- the `8M` subset is therefore a strict ordered subset of the `12M` subset

Verification notes for the prompt bank:

- `meta.seed = 1337`
- `n_train = 15000000`
- `n_val = 5000`
- `m = 21`
- `train_order.pt` is a full permutation of length `15000000`

## Seed Bookkeeping

- `bank_seed`: selects the prompt bank and therefore fixes training prompts, validation prompts, and `train_order`
- `teacher_seed`: selects which clean teacher checkpoint is used
- `render_seed`: selects rendered offline dataset identity for offline BC
- `optim.seed`: selects the training RNG for the current student or BC run

## Shared Hydra Backbone

All native online methods in this log share the same default optimizer backbone from `hydra_configs/optim/opd.yaml` unless explicitly overridden:

- `batch_size = 64`
- `learning_rate = 1e-5`
- `warmup_iters = 2000`
- `decay_lr = true`
- `lr_decay_iters = max_iters`
- `min_lr = learning_rate`
- `weight_decay = 0.0`
- `beta1 = 0.9`
- `beta2 = 0.95`
- `grad_clip = 1.0`
- `eval_interval = 5000`
- `eval_n = 5000`
- `eval_batch_size = 512`
- `log_interval = 50`
- `single_epoch = true`
- `shuffle_prompts = false`

Shared model / runtime defaults for the current S5 work:

- runtime: `gpu_float16`
- architecture source for online methods: `teacher_inferred`
- effective teacher / student architecture for this family:
  - `n_layer = 1`
  - `n_head = 8`
  - `n_embd = 512`
  - `dropout = 0.0`
  - `bias = false`

Offline BC is close but not identical:

- it uses `experiment=s5_noisy_bc`
- it runs through the `pretrain` pipeline
- it uses `hydra_configs/optim/synthetic_offline_bc.yaml`
- the main optimizer values match the online defaults above
- it uses `offline_single_epoch = true` rather than the online trainer's `single_epoch = true`

## Sweep Matrix With Seed 20260417

| Sweep | Etas | Matched law | Status | Notes |
|---|---|---|---|---|
| Offline BC | `0.0, 0.1, 0.7` | `sample_then_corrupt` | 🚫 | Interleave render and train per eta; uses rendered offline datasets rather than the prompt bank directly |
| NAIL-forward, greedy student rollout | `0.0, 0.1, 0.7` | `distributional_noise` | ✅ | Native `nail` with `loss=forward`; greedy rollout is the default NAIL behavior |
| NAIL-reverse, greedy student rollout | `0.0, 0.1, 0.7` | `distributional_noise` | ⚠️ | Native `nail` with `loss=reverse`; same MC reverse estimator as TM-OPD but on greedy student prefixes |
| NAIL-forward, sampled student rollout | `0.0, 0.1, 0.7` | `distributional_noise` | 🚫 | Same as forward NAIL except override rollout temperature to `1.0` |
| TM OPD | `0.0, 0.1, 0.7` | `distributional_noise` | 🚫 | Native `opd`; reverse-KL on sampled student rollouts |

## Sweep Matrix With Seed X

| Sweep | Etas | Matched law | Status | Notes |
|---|---|---|---|---|
| Offline BC | `0.0, 0.1, 0.7` | `sample_then_corrupt` | 🚫 | N/A |
| NAIL-forward, greedy student rollout | `0.0, 0.1, 0.7` | `distributional_noise` | 🚫 | N/A |
| NAIL-reverse, greedy student rollout | `0.0, 0.1, 0.7` | `distributional_noise` | 🚫 | N/A |
| NAIL-forward, sampled student rollout | `0.0, 0.1, 0.7` | `distributional_noise` | 🚫 | N/A |
| TM OPD | `0.0, 0.1, 0.7` | `distributional_noise` | 🚫 | N/A |

## Sweep Matrix With Seed Y

| Sweep | Etas | Matched law | Status | Notes |
|---|---|---|---|---|
| Offline BC | `0.0, 0.1, 0.7` | `sample_then_corrupt` | 🚫 | N/A |
| NAIL-forward, greedy student rollout | `0.0, 0.1, 0.7` | `distributional_noise` | 🚫 | N/A |
| NAIL-reverse, greedy student rollout | `0.0, 0.1, 0.7` | `distributional_noise` | 🚫 | N/A |
| NAIL-forward, sampled student rollout | `0.0, 0.1, 0.7` | `distributional_noise` | 🚫 | N/A |
| TM OPD | `0.0, 0.1, 0.7` | `distributional_noise` | 🚫 | N/A |

## Methods Glossary

- All methods share:
  - task family: `s5`
  - sequence setting: `m = 21`
  - prompt bank: `data/s5_clean_prompt_bank_m21_n15000000_val5000`
  - fixed prompt-bank seed: `1337`
  - fixed validation split from the prompt bank
  - strict prefix subsets from the same `train_order.pt`
  - shared depth-1 architecture family
  - shared optimizer defaults unless explicitly overridden

<details>
<summary>Offline BC</summary>

Definition:

- Train on noisy offline trajectories rendered once from the teacher.

Hydra surface:

- Experiment: `s5_noisy_bc`
- Task config family: `hydra_configs/task/s5_noisy_offline.yaml`
- Optim config family: `hydra_configs/optim/synthetic_offline_bc.yaml`
- Pipeline: `pretrain`

Current matching choice:

- use `task.rollout_mode=sample_then_corrupt`
- use `task.target_mode=tokens`
- match online `teacher_law=distributional_noise`

Important notes:

- consumes rendered datasets from disk
- depends on `task.render_seed`
- should be compared against online runs with `teacher_law=distributional_noise`

Commands:

- `[paste render/train commands here]`

Results:

- `[paste run names, metrics, or notes here]`

</details>

<details>
<summary>NAIL-forward, greedy student rollout</summary>

Definition:

- Greedy student rollout on student prefixes, then forward-KL / teacher-token CE on those prefixes.

Hydra surface:

- Experiment: `s5_nail`
- Task config family: `hydra_configs/task/s5_nail.yaml`
- Optim config family: `hydra_configs/optim/opd.yaml`
- Pipeline: `nail`

Key settings:

- `task.loss=forward`
- `task.teacher_signal=mc`
- rollout is greedy by default for NAIL
- equivalently, `task.rollout_temperature_override=0.0`

Commands:

- `[paste commands here]`

Results:

- `[paste run names, metrics, or notes here]`

</details>

<details>
<summary>NAIL-reverse, greedy student rollout</summary>

Definition:

- Greedy student rollout on student prefixes, then reverse-KL on those same prefixes.

Hydra surface:

- Experiment: `s5_nail`
- Task config family: `hydra_configs/task/s5_nail.yaml`
- Optim config family: `hydra_configs/optim/opd.yaml`
- Pipeline: `nail`

Key settings:

- `task.loss=reverse`
- `task.teacher_signal=mc`
- `task.rollout_temperature_override=0.0`

Important note:

- in the current MC setup, this uses the same sampled reverse-KL estimator as TM-OPD, but with greedy student rollouts instead of sampled student rollouts

Commands:

- `[paste commands here]`

Results:

- `[paste run names, metrics, or notes here]`

</details>

<details>
<summary>NAIL-forward, sampled student rollout</summary>

Definition:

- Same forward NAIL loss as above, but with sampled student rollouts rather than greedy ones.

Hydra surface:

- Experiment: `s5_nail`
- Task config family: `hydra_configs/task/s5_nail.yaml`
- Optim config family: `hydra_configs/optim/opd.yaml`
- Pipeline: `nail`

Key settings:

- `task.loss=forward`
- `task.teacher_signal=mc`
- `task.rollout_temperature_override=1.0`

Commands:

- `[paste commands here]`

Results:

- `[paste run names, metrics, or notes here]`

</details>

<details>
<summary>TM OPD</summary>

Definition:

- Sampled student rollout plus reverse-KL on sampled student prefixes.

Hydra surface:

- Experiment: `s5_opd`
- Task config family: `hydra_configs/task/s5_opd.yaml`
- Optim config family: `hydra_configs/optim/opd.yaml`
- Pipeline: `opd`

Key settings:

- `task.loss=reverse`
- `task.teacher_signal=mc`
- sampled student rollout is the OPD default
- equivalently, `task.rollout_temperature_override=1.0`

Commands:

- `[paste commands here]`

Results:

- `[paste run names, metrics, or notes here]`

</details>

## Terminology Glossary

<details>
<summary>Show terminology glossary</summary>

- `sample_then_corrupt`:
  offline rollout law; at each step, sample from the clean teacher distribution, then corrupt the sampled digit with probability `eta`, and feed that corrupted token into the next step
- `distributional_noise`:
  online teacher-law name for the full next-token distribution induced by `sample_then_corrupt`; this is the distribution-level counterpart of the same noisy process
- `greedy_then_corrupt`:
  offline rollout law; at each step, take the clean teacher argmax token, then corrupt that greedy digit with probability `eta`, and feed that corrupted token into the next step
- `corrupted_greedy`:
  online teacher-law name for the full next-token distribution induced by `greedy_then_corrupt`; this is the distribution-level counterpart of the greedy-corrupt process
- current matching rule:
  compare NAIL / OPD runs using `teacher_law=distributional_noise` against offline BC `sample_then_corrupt`, not against offline BC `greedy_then_corrupt`
- `teacher_signal=mc`:
  the trainer uses Monte Carlo teacher information on sampled actions / targets rather than the full teacher distribution loss
- `teacher_signal=full`:
  the trainer uses the full teacher next-token distribution in the loss
- `loss=forward`:
  forward-KL style training objective; in the MC case this becomes teacher-token cross-entropy on sampled teacher targets
- `loss=reverse`:
  reverse-KL style training objective; in the MC case this becomes the sampled reverse-KL estimator used by TM-OPD and NAIL-reverse
- `bank_seed`:
  identifies the prompt bank and therefore the fixed train prompts, validation prompts, and `train_order`
- `teacher_seed`:
  identifies the clean teacher checkpoint
- `render_seed`:
  identifies rendered offline dataset variants
- `optim.seed`:
  identifies the RNG seed for the current training run

</details>

## Open Notes

- Fill in the clean expert metrics from `reruns/s5_m21_teacher20260417/out-s5-cot-m21-depth1-seed20260417/last_eval.json`
- Fill in concrete run commands after launching each sweep family
- Fill in per-eta result summaries once runs complete
- Add explicit seed-X and seed-Y values when those sweeps are planned
