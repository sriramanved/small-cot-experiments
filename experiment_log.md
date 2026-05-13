# Experiment Guide

This is the reader-facing map from the paper experiments to the codebase. It is
not meant to be a running lab notebook or a duplicate of the paper text. Use it
after reading the paper when you want to answer practical questions like:

- Which command launches the method called NAIL-F in the paper?
- Where is the absorbing random-suffix teacher law implemented?
- Which config variable corresponds to eta, the prompt bank seed, or the
  rollout temperature?
- What files are written by a render job or by a student-prefix run?

The public entrypoint is Hydra:

```bash
python -m nanogpt.run <overrides>
```

That entrypoint is implemented in `src/nanogpt/run.py`. It materializes the
Hydra config with `src/nanogpt/config_schema.py` and dispatches to the selected
pipeline through `src/nanogpt/pipelines/__init__.py`.

## Code Map

| What a reader is looking for | Where it lives | Notes |
|---|---|---|
| Hydra entrypoint | `src/nanogpt/run.py` | Registers resolvers, loads `hydra_configs/config.yaml`, and calls `run_pipeline`. |
| Config schema / variable names | `src/nanogpt/config_schema.py` | Defines `task.eta`, `task.teacher_law`, `task.teacher_signal`, `task.loss`, seed fields, and noise configs. |
| Pipeline dispatch | `src/nanogpt/pipelines/__init__.py` | Maps `pipeline.name` to pretrain, render, prompt-bank, and student-prefix workers. |
| Clean teacher and LogLossBC trainer | `src/nanogpt/workers/pretrain_body.py` | Used by `pipeline=pretrain`; trains clean experts and LogLossBC students. |
| Student-prefix trainer | `src/nanogpt/trainers/native_student_prefix.py` | Collects student prefixes, queries the teacher, computes online losses for NAIL-F/R and OPD-F/R, and saves `run_meta.json`. |
| Online loss helpers | `src/nanogpt/methods/student_prefix.py` | Implements rollout collection, teacher distributions, forward KL, reverse KL, mixed/JSD losses, and teacher sampling. |
| Modular addition task | `data/modular_addition/task.py` | Defines tokens, prompt/target construction, corruption IDs, and clean evaluation. |
| Modular addition prompt/render pipelines | `src/nanogpt/pipelines/modadd_data.py`, `data/modular_addition/offline_render.py` | Generate prompt banks and offline noisy datasets. |
| S5 task | `data/s5_cot/task.py` | Defines S5 symbolic tokens, composition, CoT examples, and clean evaluation. |
| S5 prompt/render pipelines | `src/nanogpt/pipelines/s5_data.py`, `data/s5_cot/offline_render.py` | Generate prompt banks and offline datasets. |
| Absorbing random-suffix law | `data/synthetic/random_suffix_noise.py` | Shared stateful law for S5 and modadd render/online teacher queries. |
| S5 semantic-key noise | `data/s5_cot/semantic_key_noise.py` | Selects one semantic coordinate per S5 CoT block. |
| Offline dataset loader | `data/synthetic/offline_dataset.py` | Loads `train_x.pt`, `train_y.pt`, optional `train_teacher_probs.pt`, and validation splits. |
| Naming helpers | `src/nanogpt/utils/resolvers.py` | Defines generated prompt-bank, dataset, run, and output directory names. |
| Plot/audit helpers | `scripts/`, `notebooks/`, `debugging-log/` | Useful for analysis, but not the primary launch API. |

The attached paper also describes GSM8K experiments. This repository's
supported Hydra surface is for the synthetic S5 and modular-addition suites;
there is no checked-in GSM8K Hydra pipeline in this repo snapshot.

## Paper Names To Configs

The paper distinguishes the rollout distribution from the divergence or token
loss used on visited prefixes. The code exposes those two choices separately:

- Prefix collection is controlled by `task.rollout_temperature_override`.
  NAIL-F and NAIL-R use greedy rollout temperature `0.0`; OPD-F and OPD-R use
  sampled rollout temperature `1.0`.
- The local teacher/student comparison is controlled by `task.loss` and
  `task.teacher_signal`.

| Paper method | Hydra surface | Important overrides |
|---|---|---|
| LogLossBC / SFT | `experiment=modadd_noisy_bc` or `experiment=s5_noisy_bc` | Consumes a rendered offline dataset. Uses `pipeline=pretrain`. |
| NAIL-F | `experiment=modadd_nail` or `experiment=s5_nail` | `task.loss=forward task.teacher_signal=mc task.rollout_temperature_override=0.0` |
| NAIL-R | `experiment=modadd_nail` or `experiment=s5_nail` | `task.loss=reverse task.teacher_signal=mc task.rollout_temperature_override=0.0` |
| OPD-F | `experiment=modadd_nail` or `experiment=s5_nail` | `task.loss=forward task.teacher_signal=mc task.rollout_temperature_override=1.0`; native `opd` only supports reverse loss. |
| OPD-R | `experiment=modadd_opd` or `experiment=s5_opd` | `task.loss=reverse task.teacher_signal=mc`; rollout temperature is sampled by default. |
| Full-distribution NAIL-F or OPD-F | `task.teacher_signal=full task.loss=forward` | Uses exact teacher distributions instead of sampled teacher tokens. |
| Full-distribution NAIL-R or OPD-R | `task.teacher_signal=full task.loss=reverse` | Uses exact per-token `KL(student || teacher)`. |
| Forward/reverse interpolation | `task.loss=mixed task.teacher_signal=mc task.kl_beta=<beta>` | `beta=0` is forward-heavy; `beta=1` is reverse-heavy. |

The main online implementation is `run_student_prefix` in
`src/nanogpt/trainers/native_student_prefix.py`. In each step it:

1. Selects clean prompts from a prompt bank using `train_order.pt`.
2. Rolls out the current student with `rollout_student`.
3. Queries the frozen clean teacher on those prefixes with
   `cached_teacher_token_probs`.
4. Applies the configured noisy teacher law.
5. Updates the student with the requested forward, reverse, mixed, or JSD loss.

The code deliberately treats sampled prefixes as stopped for the gradient
update. The update is through the next-token student distribution at the visited
prefixes, matching the implementation description in the paper appendix.

## Variables

| Paper concept | Code/config variable | Notes |
|---|---|---|
| Noise level eta | `task.eta` | Used by render jobs and online teacher queries. |
| Clean expert / pi_star | `task.teacher_checkpoint` | Path to the frozen clean teacher checkpoint. |
| Noisy expert law | `task.teacher_law` | Common values: `distributional_noise`, `semantic_key_noise`, `random_suffix_after_error`. |
| Prompt distribution | prompt bank under `task.prompt_bank_dir` | Prompt banks store train prompts, validation prompts, clean CoT targets, and `train_order.pt`. |
| Training subset size | `task.subset_size` | Online runs use the first `subset_size` entries of `train_order.pt`; offline render creates the same-size dataset. |
| Prompt bank seed | `task.bank_seed` | Selects the prompt bank and the nested train order. |
| Clean teacher seed | `task.teacher_seed` and `optim.seed` during teacher training | Determines the teacher checkpoint path/name. |
| Offline render seed | `task.render_seed` | Selects the rendered offline dataset identity. |
| Random-suffix sampling seed | `task.random_suffix_noise.seed` | Defaults to the render seed for render jobs, and is often set to `optim.seed` for online bookkeeping. |
| Student training seed | `optim.seed` | Controls student initialization and training RNG. |
| Rollout temperature | `task.rollout_temperature_override` | `0.0` means greedy; `1.0` means temperature-one sampling. |
| Loss temperature | `task.loss_temperature_override` | Optional temperature for the distribution used in forward/mixed/JSD losses. |
| Monte Carlo teacher signal | `task.teacher_signal=mc` | Samples teacher actions from the noisy teacher distribution. |
| Full teacher signal | `task.teacher_signal=full` | Uses the full teacher next-token distribution. |

## Synthetic Tasks

### Modular Addition

The modular-addition task is implemented in `data/modular_addition/task.py`.
For modulus `p` and sequence length `m`:

- The vocabulary is `0, ..., p - 1, =`.
- The prompt is `m` residues followed by `=`.
- The CoT target is the length-`m` sequence of running sums modulo `p`.
- The final answer is the last running sum, so `final_answer_len = 1`.
- `prompt_len = m + 1`, `target_len = cot_len = m`, and the packed training
  block size is `2 * m`.

For the paper modular-addition suite, the important settings are:

```text
task.modadd_p = 7
task.modadd_m = 31
task.n_train = 15000000
task.n_val = 5000
task.bank_seed = 1337
task.teacher_seed = 20260417
task.subset_size = 3000000
eta in {0.0, 0.2}
seeds in {20260417, 20260418, 20260419}
```

Prompt banks are generated by `run_modadd_prompt_bank` in
`src/nanogpt/pipelines/modadd_data.py`. They contain:

```text
clean_train_prompt_ids.pt
clean_train_cot_ids.pt
clean_val_prompt_ids.pt
clean_val_cot_ids.pt
train_order.pt
meta.json
```

Offline rendered datasets are generated by `run_modadd_render` and saved under
`data/<dataset_name>/`. They contain:

```text
train_x.pt
train_y.pt
val_x.pt
val_y.pt
subset_indices.pt
clean_train_prompt_ids.pt
clean_train_cot_ids.pt
clean_val_prompt_ids.pt
clean_val_cot_ids.pt
meta.json
```

### S5

The S5 task is implemented in `data/s5_cot/task.py`. It composes permutations
of five elements using the symbolic vocabulary:

```text
( ) = 1 2 3 4 5
```

For sequence length `m`:

- Each input permutation block has 7 tokens: `(`, five values, `)`.
- The prompt contains `m` input blocks followed by `=`.
- The CoT target contains `m` running-composition blocks.
- `prompt_len = 7 * m + 1`, `target_len = cot_len = 7 * m`, and
  `final_answer_len = 7`.

S5 supports the same student-prefix trainer, plus S5-specific teacher laws:

- `semantic_key_noise`, implemented in `data/s5_cot/semantic_key_noise.py`.
- `random_suffix_after_error`, implemented through the shared
  `data/synthetic/random_suffix_noise.py` path with S5 masks from
  `data/s5_cot/offline_render.py` and `src/nanogpt/methods/student_prefix.py`.

## Teacher Laws

`distributional_noise` is the standard per-token corruption law. For offline
datasets, the matching render mode is usually `task.rollout_mode=sample_then_corrupt`:
sample the clean teacher's next token, corrupt eligible tokens with probability
`eta`, and feed the realized token into the next teacher query. For online runs,
`compute_teacher_token_probs` builds the corresponding noisy teacher
distribution at each student prefix.

`semantic_key_noise` is S5-only. It corrupts one semantic value coordinate per
CoT block, selected by `task.semantic_key_noise.coord_strategy` (`cyclic`,
`fixed`, or `hash`). Non-key positions keep the clean teacher distribution. The
eligible values are `1..5`, mapped to token IDs by
`eligible_token_ids_from_values`.

`random_suffix_after_error` is the absorbing law used for the modular-addition
paper experiment. The shared implementation is
`data/synthetic/random_suffix_noise.py`.

For modular addition:

- Every target token is semantic, so the key mask and semantic mask are all
  true. This is encoded in `data/modular_addition/offline_render.py` and in the
  modadd branch of `_random_suffix_online_masks`.
- Eligible random suffix tokens are residues `0..p-1`, from
  `data.modular_addition.task.corruptible_token_ids(p)`.
- In an unpoisoned state, the next-token distribution is
  `(1 - eta) * clean_teacher + eta * Uniform(0..p-1)`.
- If the sampled token differs from the clean running-sum token, the trajectory
  becomes poisoned.
- Once poisoned, all later semantic feedback is uniform over residues and no
  longer carries information about the clean computation.

Offline and online runs differ in where the poison state comes from:

- LogLossBC uses full noisy trajectories rendered in advance. Poisoning is sampled
  during rendering, so at `eta=0.0` the rendered dataset is clean.
- Student-prefix training does not render a full teacher trajectory. Poisoning is
  inferred from the student prefix by `compute_poisoned_before`: if any previous
  semantic student action differs from the clean target, later feedback becomes
  uniform. Thus online `eta=0.0` is still not identical to an ordinary clean
  teacher when the student has already made a prefix mistake.

For the paper setting `p=7`, `m=31`, `eta=0.2`, the idealized probability that
an offline trajectory never visibly poisons is:

```text
(1 - eta + eta / p)^m = (0.8 + 0.2 / 7)^31 ~= 0.0029
```

That is why the offline baseline is deliberately stringent in this experiment:
almost all rendered trajectories eventually contain a random suffix.

## Outputs

Training and render runs write enough metadata to reconstruct how they were
launched:

- `launcher_command.txt`: exact Hydra command.
- `launcher_config.json`: materialized config.
- `run_meta.json`: student-prefix metadata, including resolved rollout
  temperature and teacher law.
- `meta.json`: prompt-bank or offline-dataset metadata.
- `eval_history.jsonl`: evaluation curve points.
- `last_eval.json`: most recent evaluation summary.
- `ckpt.pt`: rolling checkpoint.
- `completed.txt`: written by completed student-prefix runs.

The main clean evaluation metrics are:

- `val/clean_full_exact`: full autoregressive CoT exact match.
- `val/clean_final_exact`: final-answer exact match.
- `val/loss`: clean validation cross-entropy over the target continuation.

## Modular Addition Commands

<details>
<summary>Generate the modular-addition prompt bank</summary>

```bash
python -m nanogpt.run experiment=modadd_prompt_bank \
  task.modadd_p=7 task.modadd_m=31 \
  task.n_train=15000000 task.n_val=5000 \
  task.bank_seed=1337
```

</details>

<details>
<summary>Train the modular-addition clean teacher</summary>

```bash
python -m nanogpt.run experiment=modadd_cot_p7_m31 \
  task.teacher_seed=20260417 \
  optim.seed=20260417 \
  optim.max_iters=10000 \
  optim.lr_decay_iters=10000 \
  optim.eval_interval=500
```

With the current resolvers this writes the checkpoint directory used below:
`reruns/modadd_p7_m31_teacher20260417/out-modadd-cot-p7-m31-depth1-seed20260417`.

</details>

<details>
<summary>Render modular-addition random-suffix datasets for LogLossBC</summary>

```bash
for eta in 0.0 0.2; do
  eta_tag=${eta/./p}
  for seed in 20260417 20260418 20260419; do
    python -m nanogpt.run experiment=modadd_noisy_render \
      task.modadd_p=7 task.modadd_m=31 \
      task.n_train=15000000 task.n_val=5000 \
      task.bank_seed=1337 task.teacher_seed=20260417 task.render_seed=$seed \
      task.subset_size=3000000 \
      task.prompt_bank_dir=data/modadd_clean_prompt_bank_p7_m31_n15000000_val5000 \
      task.teacher_checkpoint=reruns/modadd_p7_m31_teacher20260417/out-modadd-cot-p7-m31-depth1-seed20260417 \
      task.teacher_law=random_suffix_after_error task.eta=$eta \
      task.random_suffix_noise.seed=$seed task.random_suffix_noise.apply_to=modadd \
      task.gen_batch_size=8192
  done
done
```

</details>

<details>
<summary>Train modular-addition LogLossBC</summary>

```bash
for eta in 0.0 0.2; do
  eta_tag=${eta/./p}
  for seed in 20260417 20260418 20260419; do
    python -m nanogpt.run experiment=modadd_noisy_bc \
      task.modadd_p=7 task.modadd_m=31 \
      task.teacher_seed=20260417 task.render_seed=$seed \
      task.subset_size=3000000 \
      task.teacher_law=random_suffix_after_error task.eta=$eta \
      task.random_suffix_noise.seed=$seed task.random_suffix_noise.apply_to=modadd \
      optim.seed=$seed optim.eval_interval=500 \
      run.name=modadd-rsuffix-loglossbc-eta${eta_tag}-seed${seed} \
      logging.wandb_run_name=modadd-rsuffix-loglossbc-eta${eta_tag}-seed${seed} \
      run.out_dir=reruns/modadd_random_suffix/loglossbc/eta${eta_tag}_seed${seed}
  done
done
```

</details>

<details>
<summary>Train modular-addition NAIL-F</summary>

```bash
for eta in 0.0 0.2; do
  eta_tag=${eta/./p}
  for seed in 20260417 20260418 20260419; do
    python -m nanogpt.run experiment=modadd_nail \
      task.modadd_p=7 task.modadd_m=31 \
      task.n_train=15000000 task.n_val=5000 \
      task.bank_seed=1337 task.teacher_seed=20260417 task.render_seed=$seed \
      task.subset_size=3000000 \
      task.prompt_bank_dir=data/modadd_clean_prompt_bank_p7_m31_n15000000_val5000 \
      task.teacher_checkpoint=reruns/modadd_p7_m31_teacher20260417/out-modadd-cot-p7-m31-depth1-seed20260417 \
      task.teacher_law=random_suffix_after_error task.eta=$eta \
      task.loss=forward task.teacher_signal=mc task.rollout_temperature_override=0.0 \
      task.random_suffix_noise.seed=$seed task.random_suffix_noise.apply_to=modadd \
      optim.seed=$seed optim.eval_interval=500 \
      run.name=modadd-rsuffix-nail-f-eta${eta_tag}-seed${seed} \
      logging.wandb_run_name=modadd-rsuffix-nail-f-eta${eta_tag}-seed${seed} \
      run.out_dir=reruns/modadd_random_suffix/nail_f/eta${eta_tag}_seed${seed}
  done
done
```

</details>

<details>
<summary>Train modular-addition NAIL-R</summary>

```bash
for eta in 0.0 0.2; do
  eta_tag=${eta/./p}
  for seed in 20260417 20260418 20260419; do
    python -m nanogpt.run experiment=modadd_nail \
      task.modadd_p=7 task.modadd_m=31 \
      task.n_train=15000000 task.n_val=5000 \
      task.bank_seed=1337 task.teacher_seed=20260417 task.render_seed=$seed \
      task.subset_size=3000000 \
      task.prompt_bank_dir=data/modadd_clean_prompt_bank_p7_m31_n15000000_val5000 \
      task.teacher_checkpoint=reruns/modadd_p7_m31_teacher20260417/out-modadd-cot-p7-m31-depth1-seed20260417 \
      task.teacher_law=random_suffix_after_error task.eta=$eta \
      task.loss=reverse task.teacher_signal=mc task.rollout_temperature_override=0.0 \
      task.random_suffix_noise.seed=$seed task.random_suffix_noise.apply_to=modadd \
      optim.seed=$seed optim.eval_interval=500 \
      run.name=modadd-rsuffix-nail-r-eta${eta_tag}-seed${seed} \
      logging.wandb_run_name=modadd-rsuffix-nail-r-eta${eta_tag}-seed${seed} \
      run.out_dir=reruns/modadd_random_suffix/nail_r/eta${eta_tag}_seed${seed}
  done
done
```

</details>

<details>
<summary>Train modular-addition OPD-F</summary>

OPD-F uses the forward loss with sampled rollouts.

```bash
for eta in 0.0 0.2; do
  eta_tag=${eta/./p}
  for seed in 20260417 20260418 20260419; do
    python -m nanogpt.run experiment=modadd_nail \
      task.modadd_p=7 task.modadd_m=31 \
      task.n_train=15000000 task.n_val=5000 \
      task.bank_seed=1337 task.teacher_seed=20260417 task.render_seed=$seed \
      task.subset_size=3000000 \
      task.prompt_bank_dir=data/modadd_clean_prompt_bank_p7_m31_n15000000_val5000 \
      task.teacher_checkpoint=reruns/modadd_p7_m31_teacher20260417/out-modadd-cot-p7-m31-depth1-seed20260417 \
      task.teacher_law=random_suffix_after_error task.eta=$eta \
      task.loss=forward task.teacher_signal=mc task.rollout_temperature_override=1.0 \
      task.random_suffix_noise.seed=$seed task.random_suffix_noise.apply_to=modadd \
      optim.seed=$seed optim.eval_interval=500 \
      run.name=modadd-rsuffix-opd-f-eta${eta_tag}-seed${seed} \
      logging.wandb_run_name=modadd-rsuffix-opd-f-eta${eta_tag}-seed${seed} \
      run.out_dir=reruns/modadd_random_suffix/opd_f/eta${eta_tag}_seed${seed}
  done
done
```

</details>

<details>
<summary>Train modular-addition OPD-R</summary>

```bash
for eta in 0.0 0.2; do
  eta_tag=${eta/./p}
  for seed in 20260417 20260418 20260419; do
    python -m nanogpt.run experiment=modadd_opd \
      task.modadd_p=7 task.modadd_m=31 \
      task.n_train=15000000 task.n_val=5000 \
      task.bank_seed=1337 task.teacher_seed=20260417 task.render_seed=$seed \
      task.subset_size=3000000 \
      task.prompt_bank_dir=data/modadd_clean_prompt_bank_p7_m31_n15000000_val5000 \
      task.teacher_checkpoint=reruns/modadd_p7_m31_teacher20260417/out-modadd-cot-p7-m31-depth1-seed20260417 \
      task.teacher_law=random_suffix_after_error task.eta=$eta \
      task.loss=reverse task.teacher_signal=mc \
      task.random_suffix_noise.seed=$seed task.random_suffix_noise.apply_to=modadd \
      optim.seed=$seed optim.eval_interval=500 \
      run.name=modadd-rsuffix-opd-r-eta${eta_tag}-seed${seed} \
      logging.wandb_run_name=modadd-rsuffix-opd-r-eta${eta_tag}-seed${seed} \
      run.out_dir=reruns/modadd_random_suffix/opd_r/eta${eta_tag}_seed${seed}
  done
done
```

</details>

The online budget is one pass over the fixed prompt subset. With
`subset_size=3000000` and `batch_size=64`, the resolver in
`src/nanogpt/utils/resolvers.py` gives `ceil(3000000 / 64) = 46875`
iterations.

## S5 Endpoints And Commands

S5 uses the same Hydra entrypoint and the same student-prefix trainer. The
main endpoint names are:

```text
s5_prompt_bank
s5_cot or s5_cot_len21
s5_noisy_render
s5_noisy_bc
s5_nail
s5_opd
s5_nail_reverse_mc_fixed
s5_noisy_bc_full_dist
```

<details>
<summary>Generate an S5 prompt bank</summary>

```bash
python -m nanogpt.run experiment=s5_prompt_bank \
  task.s5_m=21 task.n_train=15000000 task.n_val=5000 task.bank_seed=1337
```

</details>

<details>
<summary>Train a clean S5 teacher</summary>

```bash
python -m nanogpt.run experiment=s5_cot_len21 \
  task.teacher_seed=20260417 optim.seed=20260417 \
  optim.max_iters=100000 optim.lr_decay_iters=100000
```

</details>

<details>
<summary>Render an S5 distributional-noise dataset for LogLossBC</summary>

```bash
python -m nanogpt.run experiment=s5_noisy_render \
  task.s5_m=21 \
  task.bank_seed=1337 task.teacher_seed=20260417 task.render_seed=20260417 \
  task.n_train=15000000 task.n_val=5000 task.subset_size=8000000 \
  task.prompt_bank_dir=data/s5_clean_prompt_bank_m21_n15000000_val5000 \
  task.teacher_checkpoint=reruns/s5_m21_teacher20260417/out-s5-cot-m21-depth1-seed20260417 \
  task.rollout_mode=sample_then_corrupt task.target_mode=tokens \
  task.teacher_law=distributional_noise task.eta=0.1 \
  task.gen_batch_size=8192
```

</details>

<details>
<summary>Train S5 LogLossBC</summary>

```bash
python -m nanogpt.run experiment=s5_noisy_bc \
  task.s5_m=21 \
  task.teacher_seed=20260417 task.render_seed=20260417 \
  task.subset_size=8000000 \
  task.rollout_mode=sample_then_corrupt task.target_mode=tokens \
  task.teacher_law=distributional_noise task.eta=0.1 \
  optim.seed=20260417
```

</details>

<details>
<summary>Run S5 NAIL-F</summary>

```bash
python -m nanogpt.run experiment=s5_nail \
  task.s5_m=21 \
  task.bank_seed=1337 task.teacher_seed=20260417 task.render_seed=20260417 \
  task.n_train=15000000 task.n_val=5000 task.subset_size=8000000 \
  task.prompt_bank_dir=data/s5_clean_prompt_bank_m21_n15000000_val5000 \
  task.teacher_checkpoint=reruns/s5_m21_teacher20260417/out-s5-cot-m21-depth1-seed20260417 \
  task.teacher_law=distributional_noise task.eta=0.1 \
  task.loss=forward task.teacher_signal=mc task.rollout_temperature_override=0.0 \
  optim.seed=20260417
```

</details>

<details>
<summary>Run S5 NAIL-R</summary>

```bash
python -m nanogpt.run experiment=s5_nail \
  task.s5_m=21 \
  task.bank_seed=1337 task.teacher_seed=20260417 task.render_seed=20260417 \
  task.n_train=15000000 task.n_val=5000 task.subset_size=8000000 \
  task.prompt_bank_dir=data/s5_clean_prompt_bank_m21_n15000000_val5000 \
  task.teacher_checkpoint=reruns/s5_m21_teacher20260417/out-s5-cot-m21-depth1-seed20260417 \
  task.teacher_law=distributional_noise task.eta=0.1 \
  task.loss=reverse task.teacher_signal=mc task.rollout_temperature_override=0.0 \
  optim.seed=20260417
```

</details>

<details>
<summary>Run S5 OPD-F</summary>

OPD-F uses the forward loss with sampled rollouts.

```bash
python -m nanogpt.run experiment=s5_nail \
  task.s5_m=21 \
  task.bank_seed=1337 task.teacher_seed=20260417 task.render_seed=20260417 \
  task.n_train=15000000 task.n_val=5000 task.subset_size=8000000 \
  task.prompt_bank_dir=data/s5_clean_prompt_bank_m21_n15000000_val5000 \
  task.teacher_checkpoint=reruns/s5_m21_teacher20260417/out-s5-cot-m21-depth1-seed20260417 \
  task.teacher_law=distributional_noise task.eta=0.1 \
  task.loss=forward task.teacher_signal=mc task.rollout_temperature_override=1.0 \
  optim.seed=20260417
```

</details>

<details>
<summary>Run S5 OPD-R</summary>

```bash
python -m nanogpt.run experiment=s5_opd \
  task.s5_m=21 \
  task.bank_seed=1337 task.teacher_seed=20260417 task.render_seed=20260417 \
  task.n_train=15000000 task.n_val=5000 task.subset_size=8000000 \
  task.prompt_bank_dir=data/s5_clean_prompt_bank_m21_n15000000_val5000 \
  task.teacher_checkpoint=reruns/s5_m21_teacher20260417/out-s5-cot-m21-depth1-seed20260417 \
  task.teacher_law=distributional_noise task.eta=0.1 \
  task.loss=reverse task.teacher_signal=mc \
  optim.seed=20260417
```

</details>

<details>
<summary>Switch S5 commands to semantic-key noise</summary>

Keep the same endpoint and add:

```bash
task.teacher_law=semantic_key_noise \
task.semantic_key_noise.coord_strategy=cyclic
```

</details>

<details>
<summary>Switch S5 commands to absorbing random-suffix noise</summary>

Keep the same endpoint and add:

```bash
task.teacher_law=random_suffix_after_error \
task.random_suffix_noise.apply_to=s5 \
task.random_suffix_noise.seed=<seed> \
task.random_suffix_noise.key_positions=semantic_key \
task.random_suffix_noise.random_suffix_mode=valid_tokens \
task.random_suffix_noise.keep_format_tokens=true \
task.random_suffix_noise.coord_strategy=cyclic
```

</details>
