# S5 Online OPD HF Backend Summary

## Context

We attempted a backend migration for student-prefix OPD-F/R on the S5 task from the current nanoGPT-style implementation to a new Hugging Face GPT-2 backend.

The goal was purely systems/performance:

- keep the OPD objectives unchanged
- keep the prompt bank, subset size, teacher checkpoint, and student architecture matched
- use Hugging Face cached decoding for student rollout and teacher prefix passes
- benchmark whether the HF implementation is faster enough to justify switching future OPD sweeps

This note summarizes what was implemented, what was validated, what benchmark was run, and why the current recommendation is to **stay on the existing nanoGPT OPD backend for now**.

## What Was Implemented

A separate HF-native student-prefix OPD-F/R path was added, without replacing the current implementation:

- new trainer: `train_opd_hf.py`
- new helper layer: `data/s5_cot/opd_hf.py`
- new sweep wrapper: `scripts/run_opd_hf_sweep.sh`

The HF path preserves the same three student-prefix OPD-F/R objectives:

- `reverse_kl_tm`
- `forward_kl_simple`
- `forward_kl_full`

and preserves the same noisy teacher laws:

- `distributional_noise`
- `corrupted_greedy`

Important implementation choices:

- student rollout uses HF cached decoding
- teacher prefix evaluation uses HF cached decoding
- the teacher is frozen and always run under `torch.no_grad()`
- the train-time student forward/backward pass uses `use_cache=False`
- compile support is optional and applied only to the student train path
- checkpoints are HF-native in a new format, separate from the current nanoGPT OPD checkpoints

## Why We Tried HF

The motivation was that Hugging Face generation and cache handling might be more optimized than our current native path.

That hypothesis was plausible because student-prefix OPD-F/R repeatedly does two autoregressive-style operations:

- student rollout on the prompt prefixes
- teacher evaluation on the student-visited prefixes

These are exactly the places where a better cache implementation might help.

## Functional Validation Performed

Before benchmarking, the HF backend was validated for correctness.

### 1. Helper-level unit tests

We added `tests/test_opd_hf.py` and verified:

- HF rollout returns the expected shapes
- HF rollout log-probs match the full forward-pass log-probs on the same sampled actions
- HF cached teacher distributions match the direct teacher-distribution computation
- teacher distributions normalize properly
- full forward KL is approximately zero when teacher and student distributions are identical
- forward-KL modes still reject `student_temperature = 0`
- HF resume metadata rejects backend mismatches

We also reran the existing OPD objective tests in `tests/test_opd_objectives.py`.

Result:

- `python3 -m unittest tests.test_opd_objectives tests.test_opd_hf`
- all `13` tests passed

### 2. Training smoke tests

We ran short synthetic smoke runs for:

- `reverse_kl_tm`
- `forward_kl_simple`
- `forward_kl_full`

We also ran:

- an HF resume smoke test
- an HF `--compile` smoke test

These runs completed successfully.

## Benchmark Design

We benchmarked the heaviest student-prefix OPD-F/R case first:

- objective: `forward_kl_full`
- teacher checkpoint: `out-s5-cot-len21-depth1-400k`
- prompt bank: `data/s5_clean_prompt_bank_m21_n15000000_val5000`
- subset size: `8,000,000`
- noise level: `eta = 0.2`
- teacher law: `distributional_noise`
- training batch size: `64`

The benchmark was designed as a short throughput probe, not a full training run.

We compared four configurations:

1. current nanoGPT backend, no compile
2. current nanoGPT backend, compile
3. HF backend, no compile
4. HF backend, compile

### What the probe actually measured

Each benchmark run was a short training job whose purpose was to estimate steady-state **training-step throughput**.

In particular, each measured step included the main student-prefix OPD-F/R work:

- student rollout on prompts
- teacher evaluation on the visited student prefixes
- objective-specific loss computation
- student backward/update step

This is the dominant repeated cost in student-prefix OPD-F/R, and it is the part most directly affected by:

- cache implementation
- backend choice
- compile behavior on the student train path

### Why regular eval was mostly disabled

Evaluation was effectively disabled during the throughput probe using a very large `eval_interval`, so the measured timings primarily reflect train-step throughput rather than evaluation overhead.

This was intentional for two reasons:

1. We wanted to isolate the dominant repeated OPD cost rather than mix it with periodic work.
2. The evaluation block includes multiple extra components that would add noise to a short benchmark:
   - clean CE evaluation
   - autoregressive clean-metric evaluation
   - checkpoint save time

If those are included in a short probe, the result becomes a mixed wall-clock benchmark of:

- training
- evaluation
- checkpoint serialization

rather than a clean comparison of the core student-prefix OPD-F/R loop.

### Does this omit something real?

Yes, but in a controlled way.

It is true that backend choice and compile can also affect evaluation speed. So this probe does **not** measure total end-to-end wall clock for a full overnight run.

Instead, it measures the part of runtime we expected to dominate:

- the repeated student-prefix OPD-F/R train step that happens every iteration

That was considered a reasonable first decision criterion because:

- evaluation happens only every `5000` iterations in the real runs
- the measured training-step slowdown for HF was large, about `36%`
- a gap that large in the dominant repeated step is unlikely to be reversed by evaluation effects alone

So the benchmark should be interpreted as:

- a **training-throughput probe**, not a full lifecycle benchmark

If desired, a second benchmark could be run later that includes realistic eval/checkpoint cadence and compares total wall clock over a longer window. We did not do that first because it is slower, noisier, and less diagnostic about where the slowdown comes from.

For each run, we summarized the average of the last `5` logged step-time values.

### How the steady-state number was computed

The benchmark jobs were not meant to run to convergence. They were short runs intended to get past startup effects and then collect several timing lines of the form:

- `iter ... time XXXXms`

For each log, we averaged the last `5` printed timing values. So the reported number is an estimate of steady-state per-log-window training time after warmup, not a final-training-time estimate.

## Benchmark Results

Measured steady-state training times:

- `nanogpt_full_c0`: `2659.48 ms`
- `nanogpt_full_c1`: `2766.29 ms`
- `hf_full_c0`: `3635.51 ms`
- `hf_full_c1`: `3623.95 ms`

Equivalent ranking:

1. nanoGPT backend, no compile
2. nanoGPT backend, compile
3. HF backend, compile
4. HF backend, no compile

## Interpretation

The result was not close.

Relative to the fastest current implementation:

- HF without compile was about `36.7%` slower than nanoGPT without compile
- HF with compile was about `36.3%` slower than nanoGPT without compile

Two additional observations matter:

- `--compile` did **not** help in this benchmark
- even after moving rollout and teacher-prefix logic to HF cached decoding, the full end-to-end training loop was still slower

So on this setup, the hypothesized HF generation/cache advantage did **not** translate into a faster student-prefix OPD-F/R trainer.

## Follow-up End-to-End Benchmarks

Because the first benchmark intentionally focused on the repeated training step rather than total wall clock, we ran a second, more realistic wall-clock benchmark.

This follow-up benchmark:

- kept the same `forward_kl_full` objective
- kept the same prompt bank, subset size, teacher checkpoint, and batch size
- used repeated evaluation during the run
- included checkpoint saves
- measured total elapsed wall-clock time with `/usr/bin/time`

Settings:

- objective: `forward_kl_full`
- teacher checkpoint: `out-s5-cot-len21-depth1-400k`
- prompt bank: `data/s5_clean_prompt_bank_m21_n15000000_val5000`
- subset size: `8,000,000`
- noise level: `eta = 0.2`
- teacher law: `distributional_noise`
- training batch size: `64`
- `max_iters = 220`
- `eval_interval = 100`
- `eval_n = 5000`
- `eval_batch_size = 512`

Measured total wall-clock times:

- `nanogpt_full_e2e`: `45.87 s`
- `hf_full_e2e`: `49.27 s`

Interpretation:

- HF was still slower end-to-end
- but the gap narrowed substantially compared with the pure train-throughput probe

Relative to nanoGPT:

- HF was about `7.4%` slower in the end-to-end benchmark

This is consistent with the idea that:

- the HF backend may recover some ground during evaluation and/or other non-train portions of the loop
- but not enough to offset the slower core training step

So the refined conclusion is:

- HF is much slower on the core repeated training step
- HF is only moderately slower on a more realistic short end-to-end wall-clock benchmark
- the current nanoGPT backend still wins overall on both criteria

## Larger End-to-End Benchmark

To reduce the chance that the first end-to-end comparison was too short or too dominated by startup/eval/checkpoint overheads, we ran a larger end-to-end benchmark with the same matched setup but a longer training window.

Settings:

- objective: `forward_kl_full`
- teacher checkpoint: `out-s5-cot-len21-depth1-400k`
- prompt bank: `data/s5_clean_prompt_bank_m21_n15000000_val5000`
- subset size: `8,000,000`
- noise level: `eta = 0.2`
- teacher law: `distributional_noise`
- training batch size: `64`
- `max_iters = 600`
- `eval_interval = 200`
- `eval_n = 5000`
- `eval_batch_size = 512`

Measured total wall-clock times:

- `nanogpt_full_e2e_v2`: `94.91 s`
- `hf_full_e2e_v2_c0`: `117.95 s`
- `hf_full_e2e_v2_c1`: `122.11 s`

Interpretation:

- HF without compile was about `24.3%` slower than nanoGPT
- HF with compile was about `28.7%` slower than nanoGPT
- compile again did not help the HF path; if anything it made it slightly worse

This larger benchmark is especially useful because it helps explain the gap between the earlier results:

- the pure train-throughput probe showed HF to be much slower
- the shorter end-to-end benchmark showed only a modest slowdown
- the larger end-to-end benchmark moved back toward the train-throughput result

That pattern makes sense. In the very short end-to-end run, fixed overheads such as:

- initial evaluation
- checkpoint saving
- startup/warmup effects

blurred the comparison. In the larger benchmark, there were many more ordinary training iterations between evals, so the slower core training loop had more time to dominate the total wall clock. That is why the larger benchmark gives a stronger and more decision-relevant signal than the short end-to-end benchmark.

So the overall empirical picture is now consistent:

- HF loses on the core repeated train step
- HF loses on a short end-to-end benchmark
- HF loses more clearly on a larger end-to-end benchmark

At this point, the practical conclusion is robust: for the tested student-prefix OPD-F/R setup, the current nanoGPT backend is the better systems choice.

## Why This Result Is Plausible

In hindsight, the result is not too surprising.

The current nanoGPT OPD path was already stronger than a naive implementation:

- it already had cached rollout support
- it already had cached teacher-prefix evaluation support
- it already used PyTorch SDPA / flash-style attention where available

So the HF backend was not replacing a slow uncached baseline. It was replacing an already-optimized native path.

That means the remaining differences are likely framework overheads and implementation details, not a simple “HF has cache, native code does not” gap.

## Decision

The current recommendation is:

- **do not switch student-prefix OPD-F/R to the HF backend for production sweeps**
- keep using the current `train_opd.py` backend
- keep `--compile` off for now, since it did not help in the benchmark we ran

This is a systems decision, not an algorithmic one.

The forward-KL and reverse-KL OPD methods themselves are still the same. The only question here was whether the HF runtime/backend improved throughput enough to justify adopting it. Based on the measured benchmark, the answer is currently **no**.

## What This Does and Does Not Prove

What it does support:

- for the tested student-prefix OPD-F/R configuration, the current nanoGPT backend is faster than the new HF backend
- switching to HF is not justified on speed grounds right now

What it does not prove:

- that HF is always slower on every possible OPD objective
- that no additional systems tuning could improve the HF path

However, `forward_kl_full` is the heaviest student-prefix OPD-F/R case and is a reasonable benchmark to use for the initial decision. We now tested it in three ways:

- a train-throughput probe
- a short end-to-end wall-clock benchmark with repeated eval/checkpoint behavior
- a larger end-to-end wall-clock benchmark with repeated eval/checkpoint behavior

HF lost all three comparisons, so there is currently no empirical speed case for migrating the main overnight sweeps.

## If We Want to Revisit This Later

The next systems ideas to try would be:

- use `torch.inference_mode()` instead of `torch.no_grad()` in rollout / teacher / eval paths
- separately test compiling the teacher decode path
- profile checkpoint-save overhead, since `save_pretrained(...)` is heavier than a plain `torch.save`
- benchmark `forward_kl_simple` as a second check, if desired

But based on the current evidence, the prudent choice is to keep the existing student-prefix OPD-F/R backend.
