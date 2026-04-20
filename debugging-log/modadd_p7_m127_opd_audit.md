# ModAdd `p=7, m=127` OPD Audit

## Scope

This audit is about the Hydra-era ModAdd sweep with:

- `p=7`
- `m=127`
- `subset_size=1_000_000`
- `seed=20260417`
- `eta in {0.0, 0.1, 0.3, 0.5, 0.7, 0.9}`

and the three compared methods:

- `Offline BC MC`
- `NAIL-OPD MC` (`forward_kl_simple`)
- `TM OPD MC` (`reverse_kl_tm`)

The audit has two parts:

1. A static code audit of what the methods actually do.
2. A runnable diagnostic script, [scripts/audit_modadd_opd_stack.py](/Users/vedsriraman/columbia/code/small-cot-experiments/nanoGPT/scripts/audit_modadd_opd_stack.py), that checks the actual run artifacts and emits machine-readable outputs.

## Status

The static code audit is complete.

The runtime audit script has now also been run on the dev node for the target `p=7, m=127` runs, and the summary file shows:

- status `ok`
- no missing runs
- no duplicate runs
- matched teacher checkpoint, prompt bank, subset size, seed, and student temperature across the compared families
- offline rollout mode `sample_then_corrupt`
- online teacher law `distributional_noise`

Command:

```bash
python3 scripts/audit_modadd_opd_stack.py \
  --root . \
  --p 7 \
  --m 127 \
  --subset-size 1000000 \
  --seed 20260417 \
  --etas 0.0 0.1 0.3 0.5 0.7 0.9
```

Outputs:

- `debugging-log/modadd_p7_m127_opd_audit_summary.json`
- `debugging-log/modadd_p7_m127_opd_audit_runs.csv`

## What The Code Actually Implements

### Offline BC MC

The current ModAdd noisy offline BC path is standard token cross-entropy on a pre-rendered noisy target sequence.

- Hydra noisy BC uses the pretrain pipeline with `offline_target_type=tokens` in [hydra_configs/experiment/modadd_noisy_bc.yaml](/Users/vedsriraman/columbia/code/small-cot-experiments/nanoGPT/hydra_configs/experiment/modadd_noisy_bc.yaml) and [src/nanogpt/workers/pretrain_body.py](/Users/vedsriraman/columbia/code/small-cot-experiments/nanoGPT/src/nanogpt/workers/pretrain_body.py).
- The offline renderer for ModAdd delegates to the generic synthetic renderer in [data/modular_addition/offline_render.py](/Users/vedsriraman/columbia/code/small-cot-experiments/nanoGPT/data/modular_addition/offline_render.py) and [data/synthetic/offline_render.py](/Users/vedsriraman/columbia/code/small-cot-experiments/nanoGPT/data/synthetic/offline_render.py).
- In `sample_then_corrupt`, the renderer:
  - samples a clean teacher token,
  - corrupts that sampled token with probability `eta`,
  - feeds the corrupted token back into the next decoding step.

So `Offline BC MC` is not a full-distribution KL objective. It is plain token imitation on one noisy Monte Carlo target sequence per prompt.

### NAIL-OPD MC (`forward_kl_simple`)

`forward_kl_simple` is **not** the full forward KL.

In [data/s5_cot/opd.py](/Users/vedsriraman/columbia/code/small-cot-experiments/nanoGPT/data/s5_cot/opd.py):

- teacher probabilities are computed on the student rollout trajectory,
- a teacher token target is sampled with `sample_teacher_actions(teacher_probs)`,
- the loss is `-log_student_target.mean()`.

Crucially, `log_teacher_target` is logged but it is not part of the optimized loss. So this objective is best described as:

- sample a noisy teacher target token from `teacher_probs`,
- then train the student to place mass on that sampled target.

That makes `forward_kl_simple` much closer to Monte Carlo teacher imitation than the name “forward KL” suggests.

### TM OPD MC (`reverse_kl_tm`)

`reverse_kl_tm` is an importance-weighted estimator on student-sampled actions.

In [data/s5_cot/opd.py](/Users/vedsriraman/columbia/code/small-cot-experiments/nanoGPT/data/s5_cot/opd.py):

- the student first rolls out actions from its own policy,
- the teacher distribution is then evaluated on those student-visited states,
- `advantage = log_teacher - log_q`,
- `importance_weight = exp(log_p - log_q.detach())`,
- the optimized loss is `-(importance_weight * advantage.detach()).mean()`.

This is qualitatively a much noisier objective than either offline token BC or `forward_kl_simple`.

## Teacher-Side Matching

The important teacher-side equivalence is:

- offline `sample_then_corrupt` <-> online `teacher_law=distributional_noise`
- offline `greedy_then_corrupt` <-> online `teacher_law=corrupted_greedy`

This comes from:

- [data/synthetic/offline_render.py](/Users/vedsriraman/columbia/code/small-cot-experiments/nanoGPT/data/synthetic/offline_render.py)
- [data/s5_cot/opd.py](/Users/vedsriraman/columbia/code/small-cot-experiments/nanoGPT/data/s5_cot/opd.py)

So the matched offline baseline for the current online `distributional_noise` runs is the `sample_then_corrupt` offline BC family, **not** the greedy-corrupt family.

One subtle but important point:

- teacher-side noise can still be matched,
- while trajectory source is still different.

In other words:

- offline BC uses teacher-generated noisy trajectories frozen into a dataset,
- OPD/NAIL use student-generated trajectories online.

That is a real difference, but it is a different axis than the teacher-law match.

## Student-Side And Eval-Side Behavior

### Student rollouts during training

Online methods use `rollout_student(...)` in [src/nanogpt/trainers/opd.py](/Users/vedsriraman/columbia/code/small-cot-experiments/nanoGPT/src/nanogpt/trainers/opd.py).

With `student_temperature=1.0`, the training trajectory is sampled, not greedy.

### Logged clean eval metrics

The clean metrics used in the plots are greedy autoregressive metrics:

- `val/clean_full_exact`
- `val/clean_final_exact`

These come from [data/synthetic/eval.py](/Users/vedsriraman/columbia/code/small-cot-experiments/nanoGPT/data/synthetic/eval.py) and the ModAdd wrapper in [data/modular_addition/task.py](/Users/vedsriraman/columbia/code/small-cot-experiments/nanoGPT/data/modular_addition/task.py).

So there is an eval mismatch:

- online training is sampled-policy,
- the main comparison plots are greedy-policy.

That does not make the plots invalid, but it can compress or distort differences between methods.

## Corruption Set Sanity Check

For ModAdd:

- corruptible ids are `0..p-1`
- the equals token id is `p`

This is defined in [data/modular_addition/task.py](/Users/vedsriraman/columbia/code/small-cot-experiments/nanoGPT/data/modular_addition/task.py).

So the corruption mechanism excludes the equals token and only corrupts digit tokens, which is the intended behavior.

## What Is Matched Vs Not Matched

| Axis | Offline BC MC | NAIL-OPD MC | TM OPD MC | Matched? |
| --- | --- | --- | --- | --- |
| Teacher-side noisy law | `sample_then_corrupt` dataset | `distributional_noise` teacher law | `distributional_noise` teacher law | Yes, if the offline run is the `-sample-` variant |
| Target type | sampled noisy tokens | sampled teacher targets | importance-weighted reverse-KL estimator | No |
| Trajectory source during training | frozen offline teacher trajectories | student rollouts | student rollouts | No |
| Student rollout temperature | not applicable during offline data creation | typically `1.0` | typically `1.0` | Yes across online runs |
| Main plotted eval | greedy clean exact match | greedy clean exact match | greedy clean exact match | Yes |

## Why Offline BC MC Could Track NAIL-OPD MC Closely

This is the main static conclusion from reading the code: the result is **not obviously a bug**.

The strongest code-level reason is that `Offline BC MC` and `forward_kl_simple` are more similar than their names make them sound.

They both effectively optimize student probability on sampled noisy teacher targets:

- Offline BC MC does this on a fixed offline dataset.
- `forward_kl_simple` does this online on student-visited states.

That means the surprising result:

- “offline BC MC tracks NAIL-OPD MC pretty closely”

is actually plausible from the current implementation.

## Runtime Findings

### 1. The compared runs are actually matched

The runtime summary confirms:

- offline rollout mode is `sample_then_corrupt`
- online teacher law is `distributional_noise`
- all runs share the same prompt bank, teacher checkpoint, subset size, and seed

So there is no evidence here that the comparison was accidentally apples-to-oranges.

### 2. Offline BC really is using the matched noisy teacher process

The offline dataset-vs-analytic teacher check reports:

- `eta=0.0`: mean TV distance `~1.9e-9`
- `eta=0.1`: mean TV distance `~0.0196`
- `eta=0.3`: mean TV distance `~0.0312`
- `eta=0.5`: mean TV distance `~0.0363`
- `eta=0.7`: mean TV distance `~0.0419`
- `eta=0.9`: mean TV distance `~0.0432`

These are small empirical-vs-analytic discrepancies for a Monte Carlo rendered dataset diagnostic, not evidence of a teacher-law bug. The main takeaway is that the offline `-sample-` dataset behaves like the intended `distributional_noise` teacher.

### 3. Greedy eval and sampled eval tell very different stories

This is the single most important runtime result.

For `eta in {0.1, 0.3, 0.5, 0.7}`:

- greedy `clean_full_exact = 1.0` for all three methods
- greedy `clean_final_exact = 1.0` for all three methods

But under sampled clean evaluation at temperature `1.0`:

- `clean_full_exact = 0.0` for all methods once `eta > 0`
- `clean_final_exact` is only about `0.10 - 0.18`

This means the current greedy clean plots are mostly measuring whether the correct token remains the mode of the learned next-token distribution. They are **not** measuring whether the sampled policy is robust over the full 127-step chain.

### 4. Offline BC MC and NAIL-OPD MC really are extremely close

The runtime gap summary shows:

- greedy Offline BC minus NAIL gaps are `0.0` almost everywhere
- sampled full-exact gaps are also `0.0` because both are always zero for `eta > 0`
- sampled final-exact gaps are small, with NAIL usually slightly better

So the “offline BC MC tracks NAIL-OPD MC closely” observation is real. It is not an artifact of missing runs or a teacher-law mismatch.

### 5. TM OPD is not failing because of exploding importance weights

This was an important hypothesis, and the runtime data rejects it.

The `reverse_kl_tm` diagnostics show:

- effective sample size ratio `= 1.0` at every tested `eta`
- importance weights are numerically `~1.0` with tiny variance

So the current implementation is **not** suffering from weight-collapse or ESS-collapse in the audited final checkpoints.

What does change with `eta` is the advantage scale:

- advantage std drops from about `0.086` at `eta=0.0`
- to about `0.006` at `eta=0.9`

So the more plausible issue is not weight explosion, but that the teacher signal becomes weak and low-contrast as noise increases.

### 6. TM OPD looks more mode-seeking than the other two methods

At `eta=0.9`:

- NAIL greedy: `clean_full_exact = 0.0`, `clean_final_exact = 0.1523`
- Offline BC greedy: `clean_full_exact = 0.0`, `clean_final_exact = 0.1367`
- TM OPD greedy: `clean_full_exact = 1.0`, `clean_final_exact = 1.0`

But under sampled eval at `eta=0.9`:

- NAIL sampled final exact: `0.1289`
- Offline BC sampled final exact: `0.1250`
- TM sampled final exact: `0.1094`

So TM OPD appears to be the most mode-seeking policy:

- best under greedy decoding at high noise
- slightly worse under sampled decoding

That is qualitatively consistent with reverse-KL behavior rather than a clear implementation bug.

## What Could Still Be Going Wrong

### 1. Plausible consequence of the objective

`forward_kl_simple` is a Monte Carlo imitation objective, not a full forward KL. If the student rollout distribution is already reasonable, it may behave much more like offline sampled-target BC than expected.

This is the leading non-bug explanation for offline BC and NAIL looking similar.

### 2. Plausible consequence of a weak reverse-KL signal, not weight instability

The runtime audit does **not** support the original ESS-collapse hypothesis.

Instead, the data suggests:

- importance weights stay at `~1`
- ESS ratio stays at `1.0`
- but the `reverse_kl_tm` advantage magnitude shrinks substantially as `eta` rises

So the more credible concern is that the reverse-KL learning signal becomes weak as the noisy teacher flattens, not that the implementation is blowing up due to unstable weights.

### 3. Comparison mismatch

The plots currently compare greedy clean eval, but the online methods are optimized using sampled rollouts at temperature `1.0`.

This mismatch is real, and the runtime results show it matters a lot:

- greedy decoding makes all three methods look nearly identical for most `eta`
- sampled decoding shows that none of the learned policies is robust over the full chain once `eta > 0`

So if the scientific question is about the quality of the learned stochastic policy, greedy `clean_full_exact` is too forgiving and too saturated to be the only headline metric.

## Current Bottom Line

Based on the code audit plus the runtime summary, the current evidence points to:

1. No obvious implementation bug in how the three compared methods are wired together.
2. A genuine objective-level similarity between Offline BC MC and `forward_kl_simple`, which explains why they track each other closely.
3. A strong greedy-vs-sampled evaluation mismatch that hides major differences in stochastic rollout behavior.
4. TM OPD behaving more like a mode-seeking policy than the other two methods, especially at very high noise.

## What The New Audit Script Checks

The script [scripts/audit_modadd_opd_stack.py](/Users/vedsriraman/columbia/code/small-cot-experiments/nanoGPT/scripts/audit_modadd_opd_stack.py) does the following:

1. Discovers the exact `p=7, m=127, n=1_000_000, seed=20260417` run family.
2. Verifies one run per method per requested `eta`, while reporting missing or duplicate runs explicitly.
3. Checks parity of:
   - teacher checkpoint family,
   - prompt-bank family,
   - subset size,
   - train seed,
   - offline rollout mode,
   - online teacher law,
   - online objective,
   - online student temperature.
4. Verifies that saved subset indices match the deterministic prompt-bank prefix produced by `select_train_subset(...)`.
5. For offline BC, compares the empirical rendered token distribution against the analytic teacher distribution on the same noisy trajectories.
6. For online methods, computes batch-level teacher/objective diagnostics on a fixed prompt batch:
   - teacher entropy and top-1 mass,
   - sampled teacher-target stats for `forward_kl_simple`,
   - `log_q`, `log_teacher`, `advantage`, importance weights, and ESS for `reverse_kl_tm`.
7. Recomputes greedy clean eval and sampled clean eval from checkpoints.

## Concrete Next Step

Run the script on blocklab where the target runs exist:

```bash
cd ~/small-cot-experiments/nanoGPT
source .venv/bin/activate

python3 scripts/audit_modadd_opd_stack.py \
  --root . \
  --p 7 \
  --m 127 \
  --subset-size 1000000 \
  --seed 20260417 \
  --etas 0.0 0.1 0.3 0.5 0.7 0.9
```

Then inspect:

- `debugging-log/modadd_p7_m127_opd_audit_summary.json`
- `debugging-log/modadd_p7_m127_opd_audit_runs.csv`

The main runtime questions to answer are:

1. Add student entropy / top-1 mass diagnostics for the three final checkpoints, especially at `eta=0.9`, to directly test the “TM is more mode-seeking” interpretation.
2. Make sampled clean-final exact a first-class plotted metric for ModAdd, because greedy clean-full exact is saturated for most of this sweep.
3. If the goal is to compare stochastic policies rather than greedy decoders, add lower-temperature sampled eval, e.g. `T=0.3` or `T=0.5`, in addition to the current `T=1.0` sampled eval.
