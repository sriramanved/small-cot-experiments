# S5 Noisy Offline BC Debug Summary

## Context

We are studying offline behavior cloning (BC) on the S5 task using a fixed prompt bank and a clean teacher checkpoint:

- Teacher checkpoint: `out-s5-cot-len21-depth1-400k`
- Prompt bank: `data/s5_clean_prompt_bank_m21_n15000000_val5000`
- Main comparison subset size: `N = 8,000,000`
- Validation split: fixed clean oracle validation prompts and clean oracle CoT targets

The surprising empirical observation is that offline BC trained on noisy teacher rollouts for moderate noise levels (`eta = 0.05` and `0.1`) performs at least as well as, and in some runs slightly better than, offline BC trained on clean teacher rollouts (`eta = 0`), even though the noisy teacher itself is very inaccurate on the clean task.

This note summarizes the tests we ran, what they show, what hypotheses have been ruled out, and what remains open.

## Main Observation

Using post-hoc evaluation on the full `5000` clean validation prompts:

- `eta = 0.0`: `clean_full_exact = 0.9952`
- `eta = 0.05`: `clean_full_exact = 0.9956`
- `eta = 0.1`: `clean_full_exact = 0.9980`
- `eta = 0.2`: `clean_full_exact = 0.9322`

So the anomaly is specifically:

- moderate noise (`0.05`, `0.1`) does not hurt, and may slightly help
- too much noise (`0.2`) clearly hurts

## Tests Run So Far

### 1. Clean teacher / HF conversion / clean offline rendering sanity check

We verified that the clean teacher and the HF-converted teacher agree, and that the clean offline dataset is rendered correctly.

Results on a `512`-prompt diagnostic slice:

- Original nanoGPT clean teacher on clean val:
  - `cot_exact = 0.9921875`
  - `clean_full_exact = 0.9921875`
  - `clean_final_exact = 0.9921875`
- HF-converted teacher on the same prompts:
  - `clean_full_exact = 0.9921875`
  - `clean_final_exact = 0.9921875`
- Original nanoGPT vs HF greedy rollout agreement:
  - `full_rollout_agreement = 1.0`
  - `final_answer_agreement = 1.0`
- Rendered clean offline dataset vs oracle clean CoTs:
  - `rendered_vs_oracle_full_exact = 0.99609375`
  - `rendered_vs_oracle_final_exact = 0.99609375`

Conclusion:

- The clean teacher checkpoint is good.
- The HF conversion is correct.
- The `eta = 0` offline renderer is not the source of the anomaly.

### 2. Noisy dataset integrity checks

We ran strict dataset diagnostics for the noisy datasets at `eta = 0.05`, `0.1`, and `0.2`.

For all three datasets, the following checks passed:

- subset indices match the prompt-bank prefix
- training prompts match the prompt bank
- clean training references match the prompt bank
- validation prompts match the prompt bank
- validation CoTs match the prompt bank
- metadata `subset_size`, `eta`, and teacher checkpoint match expectations

Conclusion:

- The noisy datasets are built from the intended prompt bank, subset, teacher checkpoint, and `eta`.
- The anomaly is not due to stale or mismatched offline datasets.

### 3. Validation source checks

We traced the offline evaluation path and confirmed that validation metrics are computed on the fixed clean validation split:

- clean validation prompts: `clean_val_prompt_ids.pt`
- clean validation oracle targets: `clean_val_cot_ids.pt`

These tensors are copied from the fixed prompt bank into each rendered offline dataset, including the noisy datasets.

Conclusion:

- Validation is on the fixed clean oracle validation set, not on noisy validation targets.
- The anomaly is not due to accidentally evaluating on noisy targets.

### 4. Train/val overlap discussion

The prompt bank generator does **not** explicitly enforce disjointness between train and val prompts. However, the S5 prompt space is astronomically large:

- each prompt is defined by `m = 21` independent permutations of `1..5`
- prompt space size is `120^21 ≈ 4.6e43`

For `15,000,000` train prompts and `5,000` validation prompts, the expected number of exact train/val collisions is approximately:

- `n_train * n_val / 120^21 ≈ 1.6e-33`

Conclusion:

- Exact disjointness is not guaranteed by construction.
- Exact overlap is astronomically unlikely.
- We have **not yet** run a realized-bank hash-based overlap audit, so this remains technically unclosed even though it is not a plausible practical explanation.

### 5. Noisy teacher targets vs clean oracle

We compared the saved noisy teacher rollouts in the datasets to the clean oracle CoTs.

For the saved `greedy_then_corrupt` datasets:

- `eta = 0.05`
  - `rendered_vs_clean_full_exact = 0.013669`
  - `rendered_vs_clean_final_exact = 0.019512`
- `eta = 0.1`
  - `rendered_vs_clean_full_exact = 0.0001556`
  - `rendered_vs_clean_final_exact = 0.0015045`
- `eta = 0.2`
  - `rendered_vs_clean_full_exact = 0.0`
  - `rendered_vs_clean_final_exact = 0.0008441`

The mismatch is mostly on digits, not punctuation:

- punctuation mismatch is only about `0.4%` to `1.0%`
- digit mismatch is about `55%` (`eta=0.05`), `66%` (`eta=0.1`), and `73%` (`eta=0.2`)

Conclusion:

- The noisy datasets are genuinely noisy.
- The anomaly is not due to the noisy teacher rollouts being accidentally close to the clean oracle.

### 6. Final checkpoint diagnostics on the clean validation set

We evaluated the final trained BC checkpoints directly on the full `5000` clean validation prompts.

Results:

- Clean BC (`eta = 0.0`):
  - `checkpoint_clean_train_oracle_loss = 0.000158`
  - `checkpoint_clean_full_exact = 0.9952`
- Noisy BC (`eta = 0.05`):
  - `checkpoint_clean_train_oracle_loss = 0.02441`
  - `checkpoint_clean_full_exact = 0.9956`
- Noisy BC (`eta = 0.1`):
  - `checkpoint_clean_train_oracle_loss = 0.04925`
  - `checkpoint_clean_full_exact = 0.9980`
- Noisy BC (`eta = 0.2`):
  - `checkpoint_clean_train_oracle_loss = 0.11062`
  - `checkpoint_clean_full_exact = 0.9322`

Interpretation:

- As `eta` increases, the trained model becomes worse at matching the clean oracle token-by-token on the training distribution.
- Nevertheless, moderate `eta` can still slightly improve clean autoregressive rollout accuracy.
- The effect disappears and reverses by `eta = 0.2`.

Conclusion:

- The anomaly is not a W&B visualization bug.
- The effect survives direct post-hoc checkpoint evaluation.

### 7. Sampled-noisy-teacher sanity check

To test whether the noisy teacher might still be accurate under sampling, we audited the `sample_then_corrupt` law:

- sample from the clean teacher distribution at each step
- corrupt the sampled digit with probability `eta`
- roll the corrupted token into the rest of the trajectory

Mean results over `10` seeds on the fixed `5000` clean val prompts:

- `eta = 0.05`
  - `full_exact ≈ 0.01348`
  - `final_exact ≈ 0.01948`
- `eta = 0.1`
  - `full_exact ≈ 0.00020`
  - `final_exact ≈ 0.00128`
- `eta = 0.2`
  - `full_exact = 0.0`
  - `final_exact ≈ 0.00104`

These numbers are almost identical to the saved `greedy_then_corrupt` dataset statistics.

Conclusion:

- Switching from greedy teacher rollout to sampled teacher rollout does **not** make the noisy teacher accurate.
- The anomaly is not explained by “the sampled noisy teacher is actually still good.”

## Hypotheses Considered and Ruled Out

### Ruled out: HF conversion bug

Reason:

- HF teacher matches the native nanoGPT teacher exactly on greedy rollout diagnostics.

### Ruled out: clean offline rendering bug

Reason:

- The clean rendered offline dataset matches the oracle clean CoTs at very high accuracy.

### Ruled out: noisy dataset mismatch / stale dataset reuse

Reason:

- strict diagnostics showed prompt-bank, subset, `eta`, and teacher-checkpoint alignment for the noisy datasets.

### Ruled out: validation accidentally uses noisy targets

Reason:

- offline evaluation uses the fixed clean validation prompts and clean oracle CoTs copied from the prompt bank.

### Ruled out: noisy teacher rollouts are accidentally still clean

Reason:

- `rendered_vs_clean_full_exact` is essentially zero by `eta = 0.1`.

### Ruled out: the sampled noisy teacher is secretly accurate

Reason:

- `sample_then_corrupt` teacher rollouts have almost identical near-zero clean accuracy to `greedy_then_corrupt`.

### Ruled out: the effect is universal across all `eta`

Reason:

- `eta = 0.2` clearly hurts final clean performance.

## Hypotheses Still Open

### Open: DART-style state-distribution coverage / recovery training

This is currently the leading conceptual explanation.

Idea:

- moderate rollout noise exposes the student to off-manifold prefixes
- BC then learns recovery behavior and becomes more robust at autoregressive rollout
- this can improve clean rollout exactness near the phase transition

This is qualitatively consistent with DART-style arguments about reducing covariate shift during imitation learning.

### Open: label denoising under symmetric corruption

Even when actions are corrupted, the clean action remains the most likely observed label under symmetric digit corruption for moderate `eta`.

Idea:

- BC can average away the symmetric corruption noise
- the student may recover the underlying clean structure while also benefiting from the wider state distribution

This could combine with the DART-style explanation above.

### Open: realized train/val overlap audit

Although overlap is astronomically unlikely, we have not yet run an exact hash-based overlap test on the realized prompt bank.

### Open: per-example error-set comparison

We have not yet compared exactly which validation examples are wrong for:

- the clean BC model
- the `eta = 0.05` BC model
- the `eta = 0.1` BC model

This would tell us whether moderate-noise BC is actually fixing specific clean-model failures or just shifting a few errors around.

### Open: corrupted-prefix recovery behavior

We have not yet directly tested whether the noisy-BC student is better at recovering from corrupted partial prefixes than the clean-BC student.

This would be a very direct test of the DART-style robustness hypothesis.

## Current Bottom Line

The current evidence strongly suggests that the surprising result is **not** caused by a simple implementation bug in:

- teacher loading
- HF conversion
- offline rendering
- dataset reuse
- validation source
- or the difference between greedy and sampled noisy-teacher rollouts

What remains plausible is a real moderate-noise effect:

- the noisy teacher itself is very poor on the clean task
- but offline BC trained on moderate-noise rollouts can still learn an excellent clean policy
- this likely reflects some combination of state-distribution coverage and denoising under symmetric corruption

At this point, the effect is real enough in the artifacts that it deserves explanation, but not yet strong enough to present as a settled scientific conclusion without the remaining audits.
