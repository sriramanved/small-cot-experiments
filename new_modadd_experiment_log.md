# New Modadd Random-Suffix Experiment Log

This log records the modular-addition experiments run for the new
`random_suffix_after_error` teacher/noise law. It is written as prompt context
for drafting an appendix experiments section, so it intentionally includes
implementation details, seed conventions, command templates, and interpretation
caveats.

## High-Level Purpose

The purpose of this modadd sweep is to test whether the new absorbing
random-suffix teacher law creates a sharper separation between offline behavior
cloning and online methods, especially NAIL-forward.

The motivating hypothesis is:

- Under diffuse non-adversarial noise, offline BC can still learn useful local
  algorithmic structure from corrupted chains of thought.
- Under `random_suffix_after_error`, once a semantic error occurs, the remaining
  semantic continuation becomes syntactically valid but semantically random.
- Offline BC should therefore receive fully useful trajectories only when there
  is no semantic corruption.
- Online methods should be less harmed because their greedy/on-policy prefixes
  become cleaner as training improves, so later teacher feedback becomes useful
  more often.

The modadd experiments are faster than the S5 experiments and were used as the
first large sanity check of this teacher law.

## Task And Data

Task:

- Task family: modular addition, abbreviated `modadd`
- Prime/modulus: `p = 7`
- Chain length: `m = 31`
- Target mode: chain-of-thought running sums
- Each prompt contains `m` input residues followed by an equals token.
- Each target contains the sequence of `m` running sums modulo `p`.
- Tokenization: these synthetic runs do not use GPT-2/BPE tokenization. They use
  the task-specific symbolic vocabulary from `data/modular_addition/task.py`;
  for `p = 7`, the vocabulary is residues `0` through `6` plus `=`, so
  `vocab_size = 8`. This is effectively character/symbol-level tokenization
  over the modular-addition alphabet.

Prompt bank:

- Common prompt bank seed: `bank_seed = 1337`
- Common prompt bank directory:
  `data/modadd_clean_prompt_bank_p7_m31_n15000000_val5000`
- Training prompts in bank: `n_train = 15000000`
- Validation prompts in bank: `n_val = 5000`
- The prompt bank is shared across all compared methods.
- This is the unsuffixed prompt bank, which by repo naming convention
  corresponds to the default bank seed `1337`.

Important correction made during setup:

- We initially considered using
  `data/modadd_clean_prompt_bank_p7_m31_n15000000_val5000_seed20260417`.
- We decided not to use it because the desired comparison fixes
  `bank_seed = 1337` across methods and seeds.
- The prompt-bank seed does not need to match the teacher seed.
- The correct prompt bank for this sweep is the unsuffixed
  `data/modadd_clean_prompt_bank_p7_m31_n15000000_val5000`.

Clean teacher:

- Fixed clean teacher seed: `teacher_seed = 20260417`
- Clean teacher checkpoint:
  `reruns/modadd_p7_m31_seed20260417/out-modadd-cot-p7-m31-depth1`
- The teacher was trained with:

```bash
python -m nanogpt.run experiment=modadd_cot_p7_m31 \
  optim.seed=20260417 \
  optim.max_iters=10000 \
  optim.lr_decay_iters=10000 \
  optim.eval_interval=500 \
  run.output_root=reruns/modadd_p7_m31_seed20260417
```

User-side verification:

- The teacher run was checked on W&B.
- The teacher achieved perfect accuracy.
- It is acceptable that `teacher_seed = 20260417` equals one of the later
  student/render sweep seeds. The teacher is fixed across the whole sweep, so
  this does not create a confound.

## Teacher Law

Teacher law name:

- `task.teacher_law=random_suffix_after_error`

This is the new absorbing random-suffix law. It is not a replacement for
existing laws such as `distributional_noise`; it is a separate teacher law.

For modadd, every target token is a semantic running-sum token. There are no S5
parenthesis scaffold tokens in the modadd target. Consequently, for modadd:

- every target position is treated as a semantic key position
- eligible semantic tokens are residues `0, 1, ..., p - 1`
- for `p = 7`, eligible tokens are `0..6`
- after poisoning, later semantic teacher probabilities are uniform over
  residues `0..6`

### Offline Rendering Semantics

Offline rendering explicitly samples a noisy teacher trajectory.

At each target position:

1. The clean teacher predicts the next running-sum token.
2. Because every modadd target position is a key semantic position, the rendered
   teacher samples from:

```text
pi_eta = (1 - eta) pi* + eta Uniform({0, ..., p - 1})
```

3. If the sampled token differs from the clean teacher token, the trajectory
   enters absorbing poisoned mode.
4. Once poisoned, all later semantic target tokens are sampled uniformly from
   `{0, ..., p - 1}`, independent of the clean computation.
5. The sampled corrupted/random tokens are rolled into subsequent teacher
   queries.

At `eta = 0.0`, offline rendering is clean in practice: the rendered teacher
samples the clean argmax at every position and never enters poisoned mode.

At `eta = 0.2`, each key token has a chance to trigger poisoning. Since the
uniform distribution includes the clean residue, the probability that a trigger
sample is non-clean at a position is approximately:

```text
eta * (1 - 1 / p) = 0.2 * (6 / 7) ~= 0.1714
```

For `m = 31` semantic positions, the no-poison probability under the idealized
independent trigger model is approximately:

```text
(1 - eta + eta / p)^m = (0.8 + 0.2 / 7)^31 ~= 0.0029
```

So for `eta = 0.2`, almost every offline rendered trajectory should eventually
enter poisoned mode. This is expected and is the point of the stress test.

### Online NAIL/OPD Semantics

Online methods do not sample a full teacher trajectory in advance. Instead, the
teacher is queried on student-generated prefixes.

For online `random_suffix_after_error`, poisoning is inferred from the student
prefix:

```text
poisoned_before_t[b] =
  any previous key position k < t where
  student_action[b, k] != clean_target[b, k]
```

For modadd, since every target position is key/semantic:

- if a student prefix has made any previous running-sum mistake, all later
  semantic teacher probabilities are uniform over residues `0..6`
- if the prefix has not yet made a running-sum mistake, the current teacher
  distribution is clean/noisy according to the mixture above

Important eta-zero caveat:

- Offline `eta = 0.0` data is clean.
- Online `eta = 0.0` under this law still becomes random after the student
  prefix makes a semantic mistake, because online poisoned mode is determined by
  prefix correctness rather than by an explicitly sampled teacher corruption.
- This is the intended online interpretation that was implemented.

This caveat matters when interpreting eta-zero online baselines: they are not
identical to a fully clean teacher if the student makes prefix mistakes.

## Seed Semantics

The sweep uses three seed values:

```text
20260417
20260418
20260419
```

Fixed across all runs:

- `bank_seed = 1337`
- `teacher_seed = 20260417`
- `teacher_checkpoint = reruns/modadd_p7_m31_seed20260417/out-modadd-cot-p7-m31-depth1`
- `prompt_bank_dir = data/modadd_clean_prompt_bank_p7_m31_n15000000_val5000`

Varied across sweep replicates:

- `render_seed in {20260417, 20260418, 20260419}`
- `task.random_suffix_noise.seed` is set equal to `render_seed`
- `optim.seed` is set equal to the same seed for the paired student run

Interpretation:

- `bank_seed` selects the prompt bank and fixes the set/order of prompts.
- `teacher_seed` selects the fixed clean expert checkpoint.
- `render_seed` selects the rendered offline dataset identity and, through the
  config default, can also seed random-suffix sampling.
- `random_suffix_noise.seed` is the actual RNG seed used by the offline
  random-suffix renderer.
- `optim.seed` controls student initialization/training RNG.

For this sweep, we intentionally vary `render_seed` across three values because
we want three independent noisy offline generated datasets, not merely three
student initializations on the same rendered data.

For online methods, `render_seed` is mostly used for run grouping/naming. Online
poisoning is not sampled from `render_seed`; it is inferred from the student
prefix. We still set `render_seed`, `random_suffix_noise.seed`, and `optim.seed`
to the same value for bookkeeping symmetry across the sweep.

## Sweep Design

Subsample size:

- `task.subset_size = 3000000`
- This is a `3M` prefix subset from the shared prompt bank.

Eta values:

- `eta = 0.0`
- `eta = 0.2`

Seeds:

- `20260417`
- `20260418`
- `20260419`

Methods:

1. Offline BC on rendered `random_suffix_after_error` datasets
2. NAIL-forward, MC teacher signal, greedy rollout
3. NAIL-reverse, MC teacher signal, greedy rollout
4. NAIL-forward, MC teacher signal, sampled rollout
5. TM OPD, reverse loss, MC teacher signal

The phrase "five methods" in this sweep refers to these five training/evaluation
conditions.

Run naming convention:

- We intentionally removed `3M` / `n3000000` from W&B names and explicit
  `run.name` values so reruns can be named flexibly.
- Rendered dataset directories still include `n3000000`, because the dataset
  resolver needs the subset size to distinguish on-disk datasets.
- Explicit run directories are placed under `reruns/modadd_random_suffix/...`.

Evaluation cadence:

- For modadd, we used `optim.eval_interval=500`.
- This replaced the coarser default `5000`.
- Evaluation uses the existing modadd evaluation utilities and validation bank.

## Shared Hydra Details

Common task overrides:

```text
task.modadd_p=7
task.modadd_m=31
task.n_train=15000000
task.n_val=5000
task.bank_seed=1337
task.teacher_seed=20260417
task.subset_size=3000000
task.prompt_bank_dir=data/modadd_clean_prompt_bank_p7_m31_n15000000_val5000
task.teacher_checkpoint=reruns/modadd_p7_m31_seed20260417/out-modadd-cot-p7-m31-depth1
task.teacher_law=random_suffix_after_error
task.random_suffix_noise.apply_to=modadd
```

Model/runtime defaults:

- Native nanoGPT model
- Depth-1 transformer inferred from teacher for online methods
- Offline BC uses `experiment=modadd_noisy_bc`
- Online methods use `experiment=modadd_nail` or `experiment=modadd_opd`
- Runtime default for these experiments is GPU float16 through Hydra

Optimizer defaults:

- Batch size: `64`
- Learning rate: `1e-5`
- Warmup iterations: `2000`
- Weight decay: `0.0`
- Adam beta1: `0.9`
- Adam beta2: `0.95`
- Gradient clip: `1.0`
- Evaluation interval override: `500`

Budget:

- Online native methods use `max_iters = epoch_steps(subset_size, batch_size)`.
- For `subset_size = 3000000` and `batch_size = 64`, this is:

```text
3000000 / 64 = 46875 iterations
```

- Offline BC uses the offline pretrain pipeline with `offline_single_epoch=true`;
  it is intended to train for one pass over the selected rendered subset.

## Render Commands

The render jobs generate offline noisy datasets. BC must wait until the
corresponding render job has completed.

### Render eta = 0.2

```bash
for seed in 20260417 20260418 20260419; do
  nohup env HYDRA_FULL_ERROR=1 python -m nanogpt.run experiment=modadd_noisy_render \
    task.modadd_p=7 task.modadd_m=31 \
    task.n_train=15000000 task.n_val=5000 \
    task.bank_seed=1337 task.teacher_seed=20260417 task.render_seed=$seed \
    task.subset_size=3000000 \
    task.prompt_bank_dir=data/modadd_clean_prompt_bank_p7_m31_n15000000_val5000 \
    task.teacher_checkpoint=reruns/modadd_p7_m31_seed20260417/out-modadd-cot-p7-m31-depth1 \
    task.teacher_law=random_suffix_after_error task.eta=0.2 \
    task.random_suffix_noise.seed=$seed task.random_suffix_noise.apply_to=modadd \
    task.gen_batch_size=8192 \
    > logs/modadd_random_suffix_render_eta0p2_seed${seed}.log 2>&1 &
done
```

Expected rendered dataset directories:

```text
data/modadd_noisy_offline_random_suffix_after_error_greedy_then_corrupt_seed20260417_p7_m31_n3000000_eta_0p2
data/modadd_noisy_offline_random_suffix_after_error_greedy_then_corrupt_seed20260418_p7_m31_n3000000_eta_0p2
data/modadd_noisy_offline_random_suffix_after_error_greedy_then_corrupt_seed20260419_p7_m31_n3000000_eta_0p2
```

### Render eta = 0.0

```bash
for seed in 20260417 20260418 20260419; do
  nohup env HYDRA_FULL_ERROR=1 python -m nanogpt.run experiment=modadd_noisy_render \
    task.modadd_p=7 task.modadd_m=31 \
    task.n_train=15000000 task.n_val=5000 \
    task.bank_seed=1337 task.teacher_seed=20260417 task.render_seed=$seed \
    task.subset_size=3000000 \
    task.prompt_bank_dir=data/modadd_clean_prompt_bank_p7_m31_n15000000_val5000 \
    task.teacher_checkpoint=reruns/modadd_p7_m31_seed20260417/out-modadd-cot-p7-m31-depth1 \
    task.teacher_law=random_suffix_after_error task.eta=0.0 \
    task.random_suffix_noise.seed=$seed task.random_suffix_noise.apply_to=modadd \
    task.gen_batch_size=8192 \
    > logs/modadd_random_suffix_render_eta0p0_seed${seed}.log 2>&1 &
done
```

Expected rendered dataset directories:

```text
data/modadd_noisy_offline_random_suffix_after_error_greedy_then_corrupt_seed20260417_p7_m31_n3000000_eta_0p0
data/modadd_noisy_offline_random_suffix_after_error_greedy_then_corrupt_seed20260418_p7_m31_n3000000_eta_0p0
data/modadd_noisy_offline_random_suffix_after_error_greedy_then_corrupt_seed20260419_p7_m31_n3000000_eta_0p0
```

Accidental duplicate note:

- The eta `0.0` render jobs were accidentally submitted twice on the dev node.
- Content-wise this should be harmless because eta-zero offline rendering is
  deterministic/clean.
- The only caveat is concurrent writes to the same dataset directory. If all
  jobs completed cleanly and the BC jobs load the datasets, the duplicate
  submission can be treated as harmless.

## Offline BC Commands

Offline BC consumes rendered datasets from disk. Therefore:

- `task.render_seed` in the BC command must match the rendered dataset seed.
- `task.eta` must match the rendered dataset eta.
- `task.subset_size` must match the rendered dataset subset size.
- `task.teacher_law` must match the rendered dataset teacher law.

### Offline BC eta = 0.2

```bash
for seed in 20260417 20260418 20260419; do
  nohup env HYDRA_FULL_ERROR=1 python -m nanogpt.run experiment=modadd_noisy_bc \
    task.modadd_p=7 task.modadd_m=31 \
    task.teacher_seed=20260417 task.render_seed=$seed \
    task.subset_size=3000000 \
    task.teacher_law=random_suffix_after_error task.eta=0.2 \
    task.random_suffix_noise.seed=$seed task.random_suffix_noise.apply_to=modadd \
    optim.seed=$seed optim.eval_interval=500 \
    run.name=modadd-rsuffix-bc-eta0p2-seed${seed} \
    logging.wandb_run_name=modadd-rsuffix-bc-eta0p2-seed${seed} \
    run.out_dir=reruns/modadd_random_suffix/offline_bc/eta0p2_seed${seed} \
    > logs/modadd_random_suffix_bc_eta0p2_seed${seed}.log 2>&1 &
done
```

Explicit run directories:

```text
reruns/modadd_random_suffix/offline_bc/eta0p2_seed20260417
reruns/modadd_random_suffix/offline_bc/eta0p2_seed20260418
reruns/modadd_random_suffix/offline_bc/eta0p2_seed20260419
```

W&B run names:

```text
modadd-rsuffix-bc-eta0p2-seed20260417
modadd-rsuffix-bc-eta0p2-seed20260418
modadd-rsuffix-bc-eta0p2-seed20260419
```

### Offline BC eta = 0.0

```bash
for seed in 20260417 20260418 20260419; do
  nohup env HYDRA_FULL_ERROR=1 python -m nanogpt.run experiment=modadd_noisy_bc \
    task.modadd_p=7 task.modadd_m=31 \
    task.teacher_seed=20260417 task.render_seed=$seed \
    task.subset_size=3000000 \
    task.teacher_law=random_suffix_after_error task.eta=0.0 \
    task.random_suffix_noise.seed=$seed task.random_suffix_noise.apply_to=modadd \
    optim.seed=$seed optim.eval_interval=500 \
    run.name=modadd-rsuffix-bc-eta0p0-seed${seed} \
    logging.wandb_run_name=modadd-rsuffix-bc-eta0p0-seed${seed} \
    run.out_dir=reruns/modadd_random_suffix/offline_bc/eta0p0_seed${seed} \
    > logs/modadd_random_suffix_bc_eta0p0_seed${seed}.log 2>&1 &
done
```

Explicit run directories:

```text
reruns/modadd_random_suffix/offline_bc/eta0p0_seed20260417
reruns/modadd_random_suffix/offline_bc/eta0p0_seed20260418
reruns/modadd_random_suffix/offline_bc/eta0p0_seed20260419
```

W&B run names:

```text
modadd-rsuffix-bc-eta0p0-seed20260417
modadd-rsuffix-bc-eta0p0-seed20260418
modadd-rsuffix-bc-eta0p0-seed20260419
```

## Online Method Commands

Online methods use the prompt bank directly. They do not consume the rendered
offline datasets. The rendered datasets are needed only for offline BC.

For all online commands:

- `task.prompt_bank_dir` is the shared prompt bank
- `task.teacher_checkpoint` is the fixed clean teacher
- `task.teacher_law=random_suffix_after_error`
- `task.random_suffix_noise.apply_to=modadd`
- `optim.eval_interval=500`
- `task.subset_size=3000000`

### NAIL-forward, greedy rollout

This is the main online method of interest for the expected separation.

```bash
for eta in 0.2 0.0; do
  eta_tag=${eta/./p}
  for seed in 20260417 20260418 20260419; do
    nohup env HYDRA_FULL_ERROR=1 python -m nanogpt.run experiment=modadd_nail \
      task.modadd_p=7 task.modadd_m=31 \
      task.n_train=15000000 task.n_val=5000 \
      task.bank_seed=1337 task.teacher_seed=20260417 task.render_seed=$seed \
      task.subset_size=3000000 \
      task.prompt_bank_dir=data/modadd_clean_prompt_bank_p7_m31_n15000000_val5000 \
      task.teacher_checkpoint=reruns/modadd_p7_m31_seed20260417/out-modadd-cot-p7-m31-depth1 \
      task.teacher_law=random_suffix_after_error \
      task.loss=forward task.teacher_signal=mc task.eta=$eta \
      task.random_suffix_noise.seed=$seed task.random_suffix_noise.apply_to=modadd \
      optim.seed=$seed optim.eval_interval=500 \
      run.name=modadd-rsuffix-nail-forward-eta${eta_tag}-seed${seed} \
      logging.wandb_run_name=modadd-rsuffix-nail-forward-eta${eta_tag}-seed${seed} \
      run.out_dir=reruns/modadd_random_suffix/nail_forward/eta${eta_tag}_seed${seed} \
      > logs/modadd_random_suffix_nail_forward_eta${eta_tag}_seed${seed}.log 2>&1 &
  done
done
```

Run directories:

```text
reruns/modadd_random_suffix/nail_forward/eta0p2_seed20260417
reruns/modadd_random_suffix/nail_forward/eta0p2_seed20260418
reruns/modadd_random_suffix/nail_forward/eta0p2_seed20260419
reruns/modadd_random_suffix/nail_forward/eta0p0_seed20260417
reruns/modadd_random_suffix/nail_forward/eta0p0_seed20260418
reruns/modadd_random_suffix/nail_forward/eta0p0_seed20260419
```

### NAIL-reverse, greedy rollout

```bash
for eta in 0.2 0.0; do
  eta_tag=${eta/./p}
  for seed in 20260417 20260418 20260419; do
    nohup env HYDRA_FULL_ERROR=1 python -m nanogpt.run experiment=modadd_nail \
      task.modadd_p=7 task.modadd_m=31 \
      task.n_train=15000000 task.n_val=5000 \
      task.bank_seed=1337 task.teacher_seed=20260417 task.render_seed=$seed \
      task.subset_size=3000000 \
      task.prompt_bank_dir=data/modadd_clean_prompt_bank_p7_m31_n15000000_val5000 \
      task.teacher_checkpoint=reruns/modadd_p7_m31_seed20260417/out-modadd-cot-p7-m31-depth1 \
      task.teacher_law=random_suffix_after_error \
      task.loss=reverse task.teacher_signal=mc task.eta=$eta \
      task.random_suffix_noise.seed=$seed task.random_suffix_noise.apply_to=modadd \
      optim.seed=$seed optim.eval_interval=500 \
      run.name=modadd-rsuffix-nail-reverse-eta${eta_tag}-seed${seed} \
      logging.wandb_run_name=modadd-rsuffix-nail-reverse-eta${eta_tag}-seed${seed} \
      run.out_dir=reruns/modadd_random_suffix/nail_reverse/eta${eta_tag}_seed${seed} \
      > logs/modadd_random_suffix_nail_reverse_eta${eta_tag}_seed${seed}.log 2>&1 &
  done
done
```

Run directories:

```text
reruns/modadd_random_suffix/nail_reverse/eta0p2_seed20260417
reruns/modadd_random_suffix/nail_reverse/eta0p2_seed20260418
reruns/modadd_random_suffix/nail_reverse/eta0p2_seed20260419
reruns/modadd_random_suffix/nail_reverse/eta0p0_seed20260417
reruns/modadd_random_suffix/nail_reverse/eta0p0_seed20260418
reruns/modadd_random_suffix/nail_reverse/eta0p0_seed20260419
```

### NAIL-forward, sampled rollout

This variant changes the NAIL rollout policy from greedy to sampled by setting:

```text
task.rollout_temperature_override=1.0
```

The loss remains forward MC.

```bash
for eta in 0.2 0.0; do
  eta_tag=${eta/./p}
  for seed in 20260417 20260418 20260419; do
    nohup env HYDRA_FULL_ERROR=1 python -m nanogpt.run experiment=modadd_nail \
      task.modadd_p=7 task.modadd_m=31 \
      task.n_train=15000000 task.n_val=5000 \
      task.bank_seed=1337 task.teacher_seed=20260417 task.render_seed=$seed \
      task.subset_size=3000000 \
      task.prompt_bank_dir=data/modadd_clean_prompt_bank_p7_m31_n15000000_val5000 \
      task.teacher_checkpoint=reruns/modadd_p7_m31_seed20260417/out-modadd-cot-p7-m31-depth1 \
      task.teacher_law=random_suffix_after_error \
      task.loss=forward task.teacher_signal=mc task.eta=$eta \
      task.rollout_temperature_override=1.0 \
      task.random_suffix_noise.seed=$seed task.random_suffix_noise.apply_to=modadd \
      optim.seed=$seed optim.eval_interval=500 \
      run.name=modadd-rsuffix-nail-forward-sampled-rollout-eta${eta_tag}-seed${seed} \
      logging.wandb_run_name=modadd-rsuffix-nail-forward-sampled-rollout-eta${eta_tag}-seed${seed} \
      run.out_dir=reruns/modadd_random_suffix/nail_forward_sampled_rollout/eta${eta_tag}_seed${seed} \
      > logs/modadd_random_suffix_nail_forward_sampled_rollout_eta${eta_tag}_seed${seed}.log 2>&1 &
  done
done
```

Run directories:

```text
reruns/modadd_random_suffix/nail_forward_sampled_rollout/eta0p2_seed20260417
reruns/modadd_random_suffix/nail_forward_sampled_rollout/eta0p2_seed20260418
reruns/modadd_random_suffix/nail_forward_sampled_rollout/eta0p2_seed20260419
reruns/modadd_random_suffix/nail_forward_sampled_rollout/eta0p0_seed20260417
reruns/modadd_random_suffix/nail_forward_sampled_rollout/eta0p0_seed20260418
reruns/modadd_random_suffix/nail_forward_sampled_rollout/eta0p0_seed20260419
```

### TM OPD

This is the native OPD reverse/MC method. In the codebase, native OPD uses the
reverse KL trajectory matching loss with sampled rollout actions.

```bash
for eta in 0.2 0.0; do
  eta_tag=${eta/./p}
  for seed in 20260417 20260418 20260419; do
    nohup env HYDRA_FULL_ERROR=1 python -m nanogpt.run experiment=modadd_opd \
      task.modadd_p=7 task.modadd_m=31 \
      task.n_train=15000000 task.n_val=5000 \
      task.bank_seed=1337 task.teacher_seed=20260417 task.render_seed=$seed \
      task.subset_size=3000000 \
      task.prompt_bank_dir=data/modadd_clean_prompt_bank_p7_m31_n15000000_val5000 \
      task.teacher_checkpoint=reruns/modadd_p7_m31_seed20260417/out-modadd-cot-p7-m31-depth1 \
      task.teacher_law=random_suffix_after_error \
      task.loss=reverse task.teacher_signal=mc task.eta=$eta \
      task.random_suffix_noise.seed=$seed task.random_suffix_noise.apply_to=modadd \
      optim.seed=$seed optim.eval_interval=500 \
      run.name=modadd-rsuffix-tm-opd-eta${eta_tag}-seed${seed} \
      logging.wandb_run_name=modadd-rsuffix-tm-opd-eta${eta_tag}-seed${seed} \
      run.out_dir=reruns/modadd_random_suffix/tm_opd/eta${eta_tag}_seed${seed} \
      > logs/modadd_random_suffix_tm_opd_eta${eta_tag}_seed${seed}.log 2>&1 &
  done
done
```

Run directories:

```text
reruns/modadd_random_suffix/tm_opd/eta0p2_seed20260417
reruns/modadd_random_suffix/tm_opd/eta0p2_seed20260418
reruns/modadd_random_suffix/tm_opd/eta0p2_seed20260419
reruns/modadd_random_suffix/tm_opd/eta0p0_seed20260417
reruns/modadd_random_suffix/tm_opd/eta0p0_seed20260418
reruns/modadd_random_suffix/tm_opd/eta0p0_seed20260419
```

## Complete Expected Run Matrix

For each eta in `{0.0, 0.2}` and each seed in
`{20260417, 20260418, 20260419}`, the intended complete matrix is:

```text
offline_bc
nail_forward
nail_reverse
nail_forward_sampled_rollout
tm_opd
```

That is:

```text
2 etas * 3 seeds * 5 methods = 30 training runs
```

In addition, rendered offline data is needed for offline BC:

```text
2 etas * 3 render seeds = 6 render jobs
```

Total jobs for the complete modadd random-suffix sweep:

```text
30 training jobs + 6 render jobs = 36 jobs
```

## Plotting Data Transfer

For plotting, the collaborator does not need the full rendered datasets or
model checkpoints unless they want to rerun evaluation. The lightweight files
needed for plotting are:

- `eval_history.jsonl`
- `last_eval.json`
- `run_meta.json`
- `launcher_command.txt`
- `launcher_config.json`
- `completed.txt`
- `wandb_state.json`

Rendered dataset metadata can also be included:

- `meta.json`
- `subset_indices.pt`

The transfer bundle was prepared from dev-node outputs using commands of the
following form.

Copy training run logs/metrics:

```bash
mkdir -p transfer/modadd_random_suffix_eta0p0_eta0p2_plot_data/reruns/modadd_random_suffix

for eta in eta0p0 eta0p2; do
  find reruns/modadd_random_suffix -path "*/${eta}_seed*" -type d | while read -r run_dir; do
    dest="transfer/modadd_random_suffix_eta0p0_eta0p2_plot_data/${run_dir}"
    mkdir -p "$dest"
    for f in eval_history.jsonl last_eval.json run_meta.json launcher_command.txt launcher_config.json completed.txt wandb_state.json; do
      [ -f "$run_dir/$f" ] && cp "$run_dir/$f" "$dest/"
    done
  done
done
```

Copy rendered dataset metadata only:

```bash
mkdir -p transfer/modadd_random_suffix_eta0p0_eta0p2_plot_data/data

for eta in 0p0 0p2; do
  for d in data/modadd_noisy_offline_random_suffix_after_error_greedy_then_corrupt_seed*_p7_m31_n3000000_eta_${eta}; do
    [ -d "$d" ] || continue
    dest="transfer/modadd_random_suffix_eta0p0_eta0p2_plot_data/$d"
    mkdir -p "$dest"
    [ -f "$d/meta.json" ] && cp "$d/meta.json" "$dest/"
    [ -f "$d/subset_indices.pt" ] && cp "$d/subset_indices.pt" "$dest/"
  done
done
```

Archive:

```bash
tar -czf modadd_random_suffix_eta0p0_eta0p2_plot_data_$(date +%Y%m%d).tar.gz \
  -C transfer modadd_random_suffix_eta0p0_eta0p2_plot_data
```

Pull archive to Mac:

```bash
scp vs2972@blocklab:~/small-cot-experiments/nanoGPT/modadd_random_suffix_eta0p0_eta0p2_plot_data_YYYYMMDD.tar.gz .
```

Replace `YYYYMMDD` with the actual date in the archive filename.

## Expected Interpretation For Paper Appendix

When writing the appendix, the experiment can be described as follows.

We fixed a clean modular-addition expert trained on `p = 7, m = 31` and used it
to define a noisy teacher law. The clean prompt bank contained `15M` training
prompts and `5000` validation prompts. All compared methods used the same prompt
bank seed `1337`, the same clean expert seed `20260417`, and the same 3M-example
training subset size. We evaluated two noise levels, `eta = 0.0` and `eta =
0.2`, across three seeds `20260417`, `20260418`, and `20260419`.

The absorbing random-suffix law differs from ordinary distributional noise in
that once a semantic mistake occurs, all later semantic labels are random valid
tokens independent of the original clean computation. For offline BC, semantic
mistakes are sampled during rendering; thus only trajectories with no sampled
semantic corruption are fully informative. For online NAIL/OPD, poisoned mode is
inferred from the student prefix, so as the student improves and avoids prefix
mistakes, the teacher remains informative for longer prefixes.

At `eta = 0.2`, the idealized no-poison probability over `31` modadd semantic
positions is approximately `(0.8 + 0.2 / 7)^31 ~= 0.0029`, so the offline
dataset is expected to contain very few fully clean/informative trajectories.
This should make offline BC a stringent baseline and is the regime where we
expect a possible online/offline separation.

At `eta = 0.0`, offline rendered data is clean. However, online
`random_suffix_after_error` still switches to random suffix feedback after a
student prefix mistake, because online poisoned mode is defined by prefix
correctness. Therefore eta-zero online runs should be interpreted as the
zero-exogenous-noise version of this absorbing-prefix teacher law, not as an
ordinary clean-teacher baseline.

The five training conditions in the sweep are offline BC, NAIL-forward with
greedy rollout, NAIL-reverse with greedy rollout, NAIL-forward with sampled
rollout, and TM OPD. All methods used evaluation interval `500`; online methods
used one pass over the 3M prompt subset, giving `46875` training iterations at
batch size `64`.
