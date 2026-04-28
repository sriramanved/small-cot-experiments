from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.s5_cot.prompt_bank import load_prompt_bank, select_train_subset
from data.s5_cot.semantic_key_noise import (
    SEMANTIC_KEY_NOISE_LAW,
    default_eligible_token_ids,
    semantic_key_mask,
    semantic_key_noise_config_from_obj,
)
from data.s5_cot.task import CORRUPTIBLE_IDS, VOCAB_SIZE, decode
from data.synthetic.offline_render import generate_teacher_targets


S5_RANDOM_FINAL_EXACT_BASELINE = 1.0 / 120.0
DEFAULT_DATA_ROOT = "data"
DEFAULT_S5_M = 21
DEFAULT_N_TRAIN = 6_000_000
DEFAULT_N_VAL = 5_000
DEFAULT_SUBSET_SIZE = 1_000_000
DEFAULT_BANK_SEED = 1337
DEFAULT_TEACHER_SEED = 1337
DEFAULT_RENDER_SEED = 1337
DEFAULT_ROLLOUT_MODE = "greedy_then_corrupt"
DEFAULT_TARGET_MODE = "tokens"
DEFAULT_TEACHER_LAW = "distributional_noise"


def _float_tag(value: object) -> str:
    return str(value).replace(".", "p").replace("-", "neg")


def _seed_suffix(seed: object, *, label: str = "seed", default: int = 1337) -> str:
    numeric_seed = int(seed)
    if numeric_seed == int(default):
        return ""
    return f"_{label}{numeric_seed}"


def s5_prompt_bank_name(
    *,
    m: int,
    n_train: int,
    n_val: int,
    bank_seed: int,
) -> str:
    return (
        f"s5_clean_prompt_bank_m{int(m)}_n{int(n_train)}_val{int(n_val)}"
        f"{_seed_suffix(bank_seed)}"
    )


def s5_dataset_prefix(
    *,
    rollout_mode: str,
    target_mode: str,
    render_seed: int,
    teacher_law: str = DEFAULT_TEACHER_LAW,
) -> str:
    prefix = "s5_noisy_offline"
    if teacher_law != DEFAULT_TEACHER_LAW:
        prefix += f"_{teacher_law.replace('-', '_')}"
    if target_mode == "teacher_probs":
        prefix += "_full_dist"
    if rollout_mode != DEFAULT_ROLLOUT_MODE:
        prefix += f"_{rollout_mode}"
    prefix += _seed_suffix(render_seed)
    return prefix


def s5_dataset_name(
    *,
    m: int,
    subset_size: int,
    eta: float,
    rollout_mode: str,
    target_mode: str,
    render_seed: int,
    teacher_law: str = DEFAULT_TEACHER_LAW,
) -> str:
    prefix = s5_dataset_prefix(
        rollout_mode=rollout_mode,
        target_mode=target_mode,
        render_seed=render_seed,
        teacher_law=teacher_law,
    )
    return f"{prefix}_m{int(m)}_n{int(subset_size)}_eta_{_float_tag(eta)}"


def decode_s5_ids(ids: torch.Tensor | list[int] | tuple[int, ...]) -> str:
    if isinstance(ids, torch.Tensor):
        values = ids.detach().cpu().to(dtype=torch.long).flatten().tolist()
    else:
        values = [int(x) for x in ids]
    return "".join(decode(values))


def s5_final_answer_ids(target_ids: torch.Tensor, final_answer_len: int = 7) -> torch.Tensor:
    if final_answer_len <= 0:
        raise ValueError(f"final_answer_len must be positive, got {final_answer_len}")
    if target_ids.size(-1) < final_answer_len:
        raise ValueError(
            f"target length {target_ids.size(-1)} is shorter than "
            f"final_answer_len={final_answer_len}"
        )
    return target_ids[..., -final_answer_len:]


def final_answer_exact(
    predicted_target_ids: torch.Tensor,
    clean_target_ids: torch.Tensor,
    *,
    final_answer_len: int = 7,
) -> torch.Tensor:
    return s5_final_answer_ids(predicted_target_ids, final_answer_len).eq(
        s5_final_answer_ids(clean_target_ids, final_answer_len)
    ).all(dim=-1)


def target_ids_from_train_y(train_y: torch.Tensor, *, prompt_len: int, target_len: int) -> torch.Tensor:
    start = int(prompt_len) - 1
    end = start + int(target_len)
    if start < 0:
        raise ValueError(f"prompt_len must be positive, got {prompt_len}")
    if train_y.size(1) < end:
        raise ValueError(
            f"train_y width={train_y.size(1)} is too short for prompt_len={prompt_len} "
            f"and target_len={target_len}"
        )
    return train_y[:, start:end]


def corruptible_mask_for_clean_targets(clean_targets: torch.Tensor) -> torch.Tensor:
    corruptible = torch.as_tensor(CORRUPTIBLE_IDS, dtype=clean_targets.dtype, device=clean_targets.device)
    return clean_targets.unsqueeze(-1).eq(corruptible.view(*([1] * clean_targets.ndim), -1)).any(dim=-1)


def _safe_fraction(num: int | float, denom: int | float) -> float:
    if denom == 0:
        return math.nan
    return float(num) / float(denom)


def _counter_to_sorted_dict(counter: Counter[int]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def compute_s5_offline_audit_metrics(
    noisy_targets: torch.Tensor,
    clean_targets: torch.Tensor,
    *,
    eta: float | None,
    final_answer_len: int = 7,
    chunk_size: int = 8192,
) -> dict[str, Any]:
    if noisy_targets.shape != clean_targets.shape:
        raise ValueError(
            f"noisy_targets shape {tuple(noisy_targets.shape)} does not match "
            f"clean_targets shape {tuple(clean_targets.shape)}"
        )
    if noisy_targets.ndim != 2:
        raise ValueError(f"expected rank-2 target tensors, got shape {tuple(noisy_targets.shape)}")

    n_rows = int(noisy_targets.size(0))
    target_len = int(noisy_targets.size(1))
    cot_len = max(0, target_len - int(final_answer_len))
    first_diff_counts: Counter[int] = Counter()
    first_diff_corruptible_counts: Counter[int] = Counter()

    total_tokens = n_rows * target_len
    total_corruptible = 0
    total_noncorruptible = 0
    token_match = 0
    corruptible_match = 0
    noncorruptible_match = 0
    full_exact_count = 0
    final_exact_count = 0
    any_final_token_diff_count = 0
    any_cot_diff_count = 0
    clean_final_despite_cot_diff_count = 0
    final_diff_after_cot_diff_count = 0
    first_diff_corruptible_total = 0
    final_corruptible_tokens = 0
    final_corruptible_match = 0
    final_tokens = n_rows * int(final_answer_len)
    final_token_match = 0

    for start in range(0, n_rows, int(chunk_size)):
        end = min(start + int(chunk_size), n_rows)
        noisy = noisy_targets[start:end]
        clean = clean_targets[start:end]
        matches = noisy.eq(clean)
        diffs = ~matches
        corruptible = corruptible_mask_for_clean_targets(clean)
        noncorruptible = ~corruptible

        token_match += int(matches.sum().item())
        total_corruptible += int(corruptible.sum().item())
        total_noncorruptible += int(noncorruptible.sum().item())
        corruptible_match += int(matches[corruptible].sum().item()) if torch.any(corruptible) else 0
        noncorruptible_match += int(matches[noncorruptible].sum().item()) if torch.any(noncorruptible) else 0

        row_full_exact = matches.all(dim=1)
        row_final_exact = matches[:, -final_answer_len:].all(dim=1)
        row_final_diff = diffs[:, -final_answer_len:].any(dim=1)
        row_cot_diff = diffs[:, :cot_len].any(dim=1) if cot_len > 0 else torch.zeros_like(row_final_diff)

        full_exact_count += int(row_full_exact.sum().item())
        final_exact_count += int(row_final_exact.sum().item())
        any_final_token_diff_count += int(row_final_diff.sum().item())
        any_cot_diff_count += int(row_cot_diff.sum().item())
        clean_final_despite_cot_diff_count += int((row_cot_diff & row_final_exact).sum().item())
        final_diff_after_cot_diff_count += int((row_cot_diff & row_final_diff).sum().item())

        final_matches = matches[:, -final_answer_len:]
        final_corruptible = corruptible[:, -final_answer_len:]
        final_token_match += int(final_matches.sum().item())
        final_corruptible_tokens += int(final_corruptible.sum().item())
        final_corruptible_match += (
            int(final_matches[final_corruptible].sum().item()) if torch.any(final_corruptible) else 0
        )

        row_has_diff = diffs.any(dim=1)
        if torch.any(row_has_diff):
            first_positions = torch.argmax(diffs.to(dtype=torch.int64), dim=1)
            diff_first_positions = first_positions[row_has_diff]
            counts = torch.bincount(diff_first_positions, minlength=target_len)
            for pos, count in enumerate(counts.tolist()):
                if count:
                    first_diff_counts[pos] += int(count)

            row_indices = torch.nonzero(row_has_diff, as_tuple=False).flatten()
            first_corruptible = corruptible[row_indices, diff_first_positions]
            first_diff_corruptible_total += int(first_corruptible.sum().item())
            corruptible_positions = diff_first_positions[first_corruptible]
            corruptible_counts = torch.bincount(corruptible_positions, minlength=target_len)
            for pos, count in enumerate(corruptible_counts.tolist()):
                if count:
                    first_diff_corruptible_counts[pos] += int(count)

    no_value_corruption_count = full_exact_count
    expected_value_change_rate = None
    if eta is not None:
        # The current corruption helper samples replacement digits uniformly from all
        # S5 digit IDs, including the original digit. So the directly observable
        # value-change rate is eta * 4/5 before any rollout effects.
        expected_value_change_rate = float(eta) * (1.0 - 1.0 / len(CORRUPTIBLE_IDS))

    metrics = {
        "num_examples": n_rows,
        "target_len": target_len,
        "final_answer_len": int(final_answer_len),
        "random_final_exact_baseline": S5_RANDOM_FINAL_EXACT_BASELINE,
        "noisy_final_exact": _safe_fraction(final_exact_count, n_rows),
        "noisy_full_exact": _safe_fraction(full_exact_count, n_rows),
        "token_match_rate": _safe_fraction(token_match, total_tokens),
        "corruptible_token_match_rate": _safe_fraction(corruptible_match, total_corruptible),
        "noncorruptible_token_match_rate": _safe_fraction(noncorruptible_match, total_noncorruptible),
        "empirical_corruption_rate_all": 1.0 - _safe_fraction(token_match, total_tokens),
        "empirical_corruption_rate_corruptible": 1.0 - _safe_fraction(corruptible_match, total_corruptible),
        "empirical_corruption_rate_noncorruptible": 1.0 - _safe_fraction(noncorruptible_match, total_noncorruptible),
        "expected_eta": None if eta is None else float(eta),
        "expected_value_change_rate_corruptible_uniform_including_original": expected_value_change_rate,
        "fraction_no_corruption": _safe_fraction(no_value_corruption_count, n_rows),
        "fraction_no_value_corruption": _safe_fraction(no_value_corruption_count, n_rows),
        "fraction_with_any_cot_token_diff": _safe_fraction(any_cot_diff_count, n_rows),
        "fraction_with_corrupted_final_answer_tokens": _safe_fraction(any_final_token_diff_count, n_rows),
        "fraction_with_clean_final_answer_despite_earlier_cot_corruption": _safe_fraction(
            clean_final_despite_cot_diff_count,
            n_rows,
        ),
        "fraction_with_clean_final_answer_given_earlier_cot_corruption": _safe_fraction(
            clean_final_despite_cot_diff_count,
            any_cot_diff_count,
        ),
        "fraction_with_final_answer_diff_after_earlier_cot_corruption": _safe_fraction(
            final_diff_after_cot_diff_count,
            n_rows,
        ),
        "final_answer_token_match_rate": _safe_fraction(final_token_match, final_tokens),
        "final_answer_corruptible_token_match_rate": _safe_fraction(
            final_corruptible_match,
            final_corruptible_tokens,
        ),
        "first_corruption_position_counts": _counter_to_sorted_dict(first_diff_counts),
        "first_corruption_position_corruptible_counts": _counter_to_sorted_dict(
            first_diff_corruptible_counts
        ),
        "fraction_first_corruption_position_is_corruptible": _safe_fraction(
            first_diff_corruptible_total,
            n_rows - no_value_corruption_count,
        ),
        "observed_over_random_final_exact": (
            _safe_fraction(final_exact_count, n_rows) / S5_RANDOM_FINAL_EXACT_BASELINE
        ),
    }
    return metrics


def compute_semantic_key_noise_audit_metrics(
    noisy_targets: torch.Tensor,
    clean_targets: torch.Tensor,
    clean_train_prompt_ids: torch.Tensor,
    *,
    eta: float | None,
    semantic_key_noise_config: dict[str, Any] | object | None,
    chunk_size: int = 8192,
) -> dict[str, Any]:
    if noisy_targets.shape != clean_targets.shape:
        raise ValueError(
            f"noisy_targets shape {tuple(noisy_targets.shape)} does not match "
            f"clean_targets shape {tuple(clean_targets.shape)}"
        )
    config = semantic_key_noise_config_from_obj(semantic_key_noise_config)
    eligible_token_ids = tuple(
        int(x)
        for x in (
            config.to_dict().get("eligible_token_ids")
            if hasattr(config, "to_dict")
            else default_eligible_token_ids()
        )
    )
    k = len(eligible_token_ids)
    n_rows = int(noisy_targets.size(0))
    target_len = int(noisy_targets.size(1))

    key_positions_per_row: list[int] = []
    first_key_diff_counts: Counter[int] = Counter()
    total_key_positions = 0
    total_non_key_positions = 0
    key_mismatch_count = 0
    non_key_mismatch_count = 0
    all_key_match_count = 0
    any_key_diff_count = 0
    first_key_diff_total = 0
    baseline_sum = 0.0

    for start in range(0, n_rows, int(chunk_size)):
        end = min(start + int(chunk_size), n_rows)
        noisy = noisy_targets[start:end]
        clean = clean_targets[start:end]
        prompts = clean_train_prompt_ids[start:end]
        matches = noisy.eq(clean)
        diffs = ~matches
        keys = semantic_key_mask(prompts, target_len, config)
        non_keys = ~keys
        key_diffs = diffs & keys

        row_key_counts = keys.sum(dim=1)
        key_positions_per_row.extend(int(x) for x in row_key_counts.tolist())
        total_key_positions += int(keys.sum().item())
        total_non_key_positions += int(non_keys.sum().item())
        key_mismatch_count += int(key_diffs.sum().item())
        non_key_mismatch_count += int((diffs & non_keys).sum().item())

        row_has_key_diff = key_diffs.any(dim=1)
        any_key_diff_count += int(row_has_key_diff.sum().item())
        all_key_match_count += int((~row_has_key_diff).sum().item())
        if eta is not None and k > 0:
            clean_key_prob = 1.0 - float(eta) + float(eta) / float(k)
            baseline_sum += float(torch.pow(
                torch.full_like(row_key_counts, clean_key_prob, dtype=torch.float64),
                row_key_counts.to(dtype=torch.float64),
            ).sum().item())

        if torch.any(row_has_key_diff):
            first_positions = torch.argmax(key_diffs.to(dtype=torch.int64), dim=1)
            diff_first_positions = first_positions[row_has_key_diff]
            first_key_diff_total += int(diff_first_positions.numel())
            counts = torch.bincount(diff_first_positions, minlength=target_len)
            for pos, count in enumerate(counts.tolist()):
                if count:
                    first_key_diff_counts[pos] += int(count)

    key_count_counter = Counter(key_positions_per_row)
    key_count_mean = _safe_fraction(sum(key_positions_per_row), len(key_positions_per_row))
    return {
        "enabled": True,
        "coord_strategy": config.coord_strategy,
        "eligible_token_ids": list(eligible_token_ids),
        "eligible_value_count": k,
        "key_positions_per_trajectory": {
            "min": min(key_positions_per_row) if key_positions_per_row else 0,
            "max": max(key_positions_per_row) if key_positions_per_row else 0,
            "mean": key_count_mean,
            "counts": _counter_to_sorted_dict(key_count_counter),
        },
        "total_key_positions": total_key_positions,
        "semantic_key_token_mismatch_rate": _safe_fraction(key_mismatch_count, total_key_positions),
        "empirical_key_token_corruption_or_mismatch_rate": _safe_fraction(
            key_mismatch_count,
            total_key_positions,
        ),
        "empirical_non_key_direct_corruption_or_drift_rate": _safe_fraction(
            non_key_mismatch_count,
            total_non_key_positions,
        ),
        "first_key_corruption_position_counts": _counter_to_sorted_dict(first_key_diff_counts),
        "fraction_with_any_key_token_mismatch": _safe_fraction(any_key_diff_count, n_rows),
        "fraction_all_key_tokens_matching_clean": _safe_fraction(all_key_match_count, n_rows),
        "expected_all_key_clean_rate_uniform_including_clean": (
            None if eta is None or k <= 0 else _safe_fraction(baseline_sum, n_rows)
        ),
        "first_key_corruption_observed_count": first_key_diff_total,
        "notes": {
            "non_key_rate": (
                "This is measured as non-key mismatch against the clean oracle. "
                "Rendered datasets do not save direct Bernoulli draws, so downstream "
                "drift after rolled-in key corruptions is included."
            ),
            "baseline": "(1 - eta + eta / K) ** num_key_positions, averaged over rows",
        },
    }


def example_record(
    *,
    row: int,
    prompt_ids: torch.Tensor,
    clean_target_ids: torch.Tensor,
    noisy_target_ids: torch.Tensor,
    final_answer_len: int,
    key_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    clean = clean_target_ids.detach().cpu().to(dtype=torch.long)
    noisy = noisy_target_ids.detach().cpu().to(dtype=torch.long)
    prompt = prompt_ids.detach().cpu().to(dtype=torch.long)
    matches = noisy.eq(clean)
    diffs = ~matches
    corruptible = corruptible_mask_for_clean_targets(clean.unsqueeze(0)).squeeze(0)
    diff_positions = torch.nonzero(diffs, as_tuple=False).flatten().tolist()
    key_positions: list[int] = []
    key_differences: list[int] = []
    if key_mask is not None:
        keys = key_mask.detach().cpu().to(dtype=torch.bool).flatten()
        key_positions = torch.nonzero(keys, as_tuple=False).flatten().tolist()
        key_differences = [int(pos) for pos in key_positions if bool(diffs[pos].item())]
    target_len = int(clean.numel())
    final_start = max(0, target_len - int(final_answer_len))

    differences = []
    for pos in diff_positions:
        differences.append(
            {
                "position": int(pos),
                "clean_id": int(clean[pos].item()),
                "noisy_id": int(noisy[pos].item()),
                "clean_token": decode_s5_ids([int(clean[pos].item())]),
                "noisy_token": decode_s5_ids([int(noisy[pos].item())]),
                "corruptible": bool(corruptible[pos].item()),
                "in_final_answer": bool(pos >= final_start),
            }
        )

    first_diff = int(diff_positions[0]) if diff_positions else None
    suffix = None
    if first_diff is not None:
        suffix = {
            "first_difference_position": first_diff,
            "clean_suffix_from_first_difference": decode_s5_ids(clean[first_diff:]),
            "noisy_suffix_from_first_difference": decode_s5_ids(noisy[first_diff:]),
        }

    return {
        "row": int(row),
        "prompt": decode_s5_ids(prompt),
        "clean_final_answer": decode_s5_ids(s5_final_answer_ids(clean, final_answer_len)),
        "noisy_final_answer": decode_s5_ids(s5_final_answer_ids(noisy, final_answer_len)),
        "final_answer_exact": bool(final_answer_exact(noisy.unsqueeze(0), clean.unsqueeze(0), final_answer_len=final_answer_len)[0].item()),
        "clean_target": decode_s5_ids(clean),
        "noisy_target": decode_s5_ids(noisy),
        "num_differences": len(differences),
        "differences": differences,
        "semantic_key_positions": key_positions,
        "semantic_key_difference_positions": key_differences,
        "first_semantic_key_corruption_position": (
            int(key_differences[0]) if key_differences else None
        ),
        "rollout_suffix_after_first_difference": suffix,
    }


def build_examples(
    *,
    clean_train_prompt_ids: torch.Tensor,
    clean_targets: torch.Tensor,
    noisy_targets: torch.Tensor,
    final_answer_len: int,
    num_examples: int,
    start_row: int = 0,
    key_masks: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    examples = []
    start_row = max(0, int(start_row))
    end_row = min(int(clean_targets.size(0)), start_row + max(0, int(num_examples)))
    for row in range(start_row, end_row):
        examples.append(
            example_record(
                row=row,
                prompt_ids=clean_train_prompt_ids[row],
                clean_target_ids=clean_targets[row],
                noisy_target_ids=noisy_targets[row],
                final_answer_len=final_answer_len,
                key_mask=None if key_masks is None else key_masks[row],
            )
        )
    return examples


def renderer_rollin_static_check() -> dict[str, Any]:
    source = inspect.getsource(generate_teacher_targets)
    corrupt_pos = source.find("next_ids = corrupt_ids_fn")
    prob_rollout_pos = source.find("rollout_probs_step_fn")
    feed_pos = source.find("input_ids = next_ids.unsqueeze(1)")
    return {
        "renderer_function": "data.synthetic.offline_render.generate_teacher_targets",
        "corrupts_before_saving_token": corrupt_pos >= 0,
        "supports_per_step_probability_rollout": prob_rollout_pos >= 0,
        "feeds_next_query_from_post_corruption_token": feed_pos >= 0 and (corrupt_pos < 0 or feed_pos > corrupt_pos),
        "feeds_next_query_from_sampled_probability_rollout_token": (
            feed_pos >= 0 and prob_rollout_pos >= 0 and feed_pos > prob_rollout_pos
        ),
        "evidence": (
            "The renderer either applies corrupt_ids_fn to next_ids or samples next_ids "
            "from rollout_probs_step_fn, stores that token in generated[:, step], then "
            "sets input_ids = next_ids.unsqueeze(1) before the next model call."
        ),
    }


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bool_check(name: str, value: bool | None) -> dict[str, Any]:
    return {"name": name, "ok": value}


def clean_validation_consistency_report(
    *,
    dataset_dir: Path,
    prompt_bank_dir: Path,
    meta: dict[str, Any],
    prompt_len: int,
    target_len: int,
    final_answer_len: int,
) -> dict[str, Any]:
    prompt_bank = load_prompt_bank(prompt_bank_dir)

    checks: list[dict[str, Any]] = []

    subset_path = dataset_dir / "subset_indices.pt"
    if subset_path.exists():
        saved_subset_idx = torch.load(subset_path, map_location="cpu").long()
        subset_size = int(meta.get("subset_size", saved_subset_idx.numel()))
        expected_subset_idx = select_train_subset(prompt_bank, subset_size)
        checks.append(bool_check("subset_indices_match_prompt_bank_order", torch.equal(saved_subset_idx, expected_subset_idx)))
    else:
        saved_train_prompt_for_size = torch.load(dataset_dir / "clean_train_prompt_ids.pt", map_location="cpu")
        subset_size = int(meta.get("subset_size", saved_train_prompt_for_size.size(0)))
        expected_subset_idx = select_train_subset(prompt_bank, subset_size)
        saved_subset_idx = expected_subset_idx
        checks.append(bool_check("subset_indices_present", False))

    saved_train_prompt = torch.load(dataset_dir / "clean_train_prompt_ids.pt", map_location="cpu")
    saved_train_cot = torch.load(dataset_dir / "clean_train_cot_ids.pt", map_location="cpu")
    saved_val_prompt = torch.load(dataset_dir / "clean_val_prompt_ids.pt", map_location="cpu")
    saved_val_cot = torch.load(dataset_dir / "clean_val_cot_ids.pt", map_location="cpu")

    expected_train_prompt = prompt_bank.clean_train_prompt_ids.index_select(0, saved_subset_idx)
    expected_train_cot = prompt_bank.clean_train_cot_ids.index_select(0, saved_subset_idx)
    checks.append(bool_check("saved_clean_train_prompts_match_prompt_bank_subset", torch.equal(saved_train_prompt, expected_train_prompt)))
    checks.append(bool_check("saved_clean_train_targets_match_prompt_bank_subset", torch.equal(saved_train_cot, expected_train_cot)))
    checks.append(bool_check("saved_clean_val_prompts_match_prompt_bank", torch.equal(saved_val_prompt, prompt_bank.clean_val_prompt_ids)))
    checks.append(bool_check("saved_clean_val_targets_match_prompt_bank", torch.equal(saved_val_cot, prompt_bank.clean_val_cot_ids)))

    val_y = torch.load(dataset_dir / "val_y.pt", map_location="cpu")
    val_y_targets = target_ids_from_train_y(val_y, prompt_len=prompt_len, target_len=target_len)
    checks.append(bool_check("offline_bc_val_y_targets_are_clean_prompt_bank_targets", torch.equal(val_y_targets.to(dtype=saved_val_cot.dtype), saved_val_cot)))

    checks.append(bool_check("final_answer_tokens_in_validation_target", target_len >= final_answer_len and final_answer_len > 0))

    return {
        "offline_bc_validation_target_source": meta.get("val_targets_source", "fixed_clean_oracle"),
        "offline_bc_train_target_source": meta.get("train_targets_source", "teacher_rollout_with_optional_eta_corruption"),
        "opd_nail_validation_target_source": (
            "prompt_bank.clean_val_prompt_ids and prompt_bank.clean_val_cot_ids via "
            "evaluate_clean_ce_loss/evaluate_saved_clean_s5_metrics"
        ),
        "opd_nail_training_prompt_source": (
            "prompt_bank.clean_train_prompt_ids selected by select_train_subset; "
            "teacher supervision is computed online on student rollouts"
        ),
        "same_clean_prompt_bank": all(check["ok"] is True for check in checks[:4]),
        "target_span": meta.get("target_span", prompt_bank.meta.get("target_span", "cot_with_final_answer_suffix")),
        "final_answer_len": final_answer_len,
        "target_len": target_len,
        "checks": checks,
    }


def flatten_metrics_for_csv(report: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key in ("eta", "dataset_dir", "prompt_bank_dir"):
        row[key] = report.get(key)
    for key, value in report["metrics"].items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            row[key] = value
    semantic = report.get("semantic_key_noise_metrics")
    if semantic is not None:
        for key, value in semantic.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                row[f"semantic_{key}"] = value
    seeds = report.get("seeds", {})
    for key, value in seeds.items():
        row[f"seed_{key}"] = value
    return row


def save_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "null":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_overrides(overrides: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"Unsupported argument {item!r}; expected key=value Hydra-style override")
        key, value = item.split("=", 1)
        key = key.lstrip("+")
        parsed[key] = parse_scalar(value)
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit rendered S5 noisy offline BC targets against the clean oracle targets "
            "for the same prompts. This does not evaluate a model."
        )
    )
    parser.add_argument("--data-dir", "--dataset-dir", dest="data_dir", default=None)
    parser.add_argument("--prompt-bank-dir", default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--examples-start", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--csv-out", default=None)
    parser.add_argument("--no-json", action="store_true")
    parser.add_argument("--no-examples", action="store_true")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def resolved_config(args: argparse.Namespace) -> dict[str, Any]:
    overrides = parse_overrides(args.overrides)
    data_dir_explicit = args.data_dir is not None
    prompt_bank_dir_explicit = args.prompt_bank_dir is not None
    cfg: dict[str, Any] = {
        "data_root": DEFAULT_DATA_ROOT,
        "s5_m": DEFAULT_S5_M,
        "n_train": DEFAULT_N_TRAIN,
        "n_val": DEFAULT_N_VAL,
        "subset_size": DEFAULT_SUBSET_SIZE,
        "bank_seed": DEFAULT_BANK_SEED,
        "teacher_seed": DEFAULT_TEACHER_SEED,
        "render_seed": DEFAULT_RENDER_SEED,
        "eta": args.eta,
        "rollout_mode": DEFAULT_ROLLOUT_MODE,
        "target_mode": DEFAULT_TARGET_MODE,
        "teacher_law": DEFAULT_TEACHER_LAW,
        "semantic_key_noise": {
            "enabled": True,
            "coord_strategy": "cyclic",
            "fixed_coord": 0,
            "seed": DEFAULT_RENDER_SEED,
            "include_clean_value": True,
            "eligible_values": [1, 2, 3, 4, 5],
            "apply_to": "partial_perm_image",
            "one_key_per_block": True,
        },
        "dataset": None,
        "data_dir": args.data_dir,
        "prompt_bank_dir": args.prompt_bank_dir,
        "num_examples": 20 if args.num_examples is None else int(args.num_examples),
        "examples_start": int(args.examples_start),
        "max_rows": args.max_rows,
        "chunk_size": int(args.chunk_size),
        "output_dir": args.output_dir,
        "json_out": args.json_out,
        "csv_out": args.csv_out,
        "no_json": bool(args.no_json),
        "no_examples": bool(args.no_examples),
        "data_dir_explicit": data_dir_explicit,
        "prompt_bank_dir_explicit": prompt_bank_dir_explicit,
    }
    mapping = {
        "task.data_root": "data_root",
        "task.s5_m": "s5_m",
        "task.n_train": "n_train",
        "task.n_val": "n_val",
        "task.subset_size": "subset_size",
        "task.bank_seed": "bank_seed",
        "task.teacher_seed": "teacher_seed",
        "task.render_seed": "render_seed",
        "task.eta": "eta",
        "task.rollout_mode": "rollout_mode",
        "task.target_mode": "target_mode",
        "task.teacher_law": "teacher_law",
        "task.dataset": "dataset",
        "task.prompt_bank_dir": "prompt_bank_dir",
        "task.semantic_key_noise.enabled": "semantic_key_noise.enabled",
        "task.semantic_key_noise.coord_strategy": "semantic_key_noise.coord_strategy",
        "task.semantic_key_noise.fixed_coord": "semantic_key_noise.fixed_coord",
        "task.semantic_key_noise.seed": "semantic_key_noise.seed",
        "task.semantic_key_noise.include_clean_value": "semantic_key_noise.include_clean_value",
        "task.semantic_key_noise.eligible_values": "semantic_key_noise.eligible_values",
        "task.semantic_key_noise.apply_to": "semantic_key_noise.apply_to",
        "task.semantic_key_noise.one_key_per_block": "semantic_key_noise.one_key_per_block",
        "audit.data_dir": "data_dir",
        "audit.dataset_dir": "data_dir",
        "audit.prompt_bank_dir": "prompt_bank_dir",
        "audit.num_examples": "num_examples",
        "audit.examples_start": "examples_start",
        "audit.max_rows": "max_rows",
        "audit.chunk_size": "chunk_size",
        "audit.output_dir": "output_dir",
        "audit.json_out": "json_out",
        "audit.csv_out": "csv_out",
        "audit.no_json": "no_json",
        "audit.no_examples": "no_examples",
    }
    for key, value in overrides.items():
        if key not in mapping:
            raise SystemExit(f"Unsupported override {key!r}")
        target_key = mapping[key]
        if target_key.startswith("semantic_key_noise."):
            nested_key = target_key.split(".", 1)[1]
            cfg["semantic_key_noise"][nested_key] = value
        else:
            cfg[target_key] = value
        if target_key == "data_dir":
            cfg["data_dir_explicit"] = True
        if target_key == "prompt_bank_dir":
            cfg["prompt_bank_dir_explicit"] = True

    if cfg["eta"] is None and not cfg["data_dir"] and not cfg["dataset"]:
        raise SystemExit("Provide --eta/task.eta, --data-dir, or task.dataset.")

    if cfg["dataset"] is None and cfg["eta"] is not None:
        cfg["dataset"] = s5_dataset_name(
            m=int(cfg["s5_m"]),
            subset_size=int(cfg["subset_size"]),
            eta=float(cfg["eta"]),
            rollout_mode=str(cfg["rollout_mode"]),
            target_mode=str(cfg["target_mode"]),
            render_seed=int(cfg["render_seed"]),
            teacher_law=str(cfg["teacher_law"]),
        )
    if cfg["data_dir"] is None:
        cfg["data_dir"] = str(Path(str(cfg["data_root"])) / str(cfg["dataset"]))
    if cfg["prompt_bank_dir"] is None:
        cfg["prompt_bank_dir"] = str(
            Path(str(cfg["data_root"]))
            / s5_prompt_bank_name(
                m=int(cfg["s5_m"]),
                n_train=int(cfg["n_train"]),
                n_val=int(cfg["n_val"]),
                bank_seed=int(cfg["bank_seed"]),
            )
        )
    return cfg


def audit_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    dataset_dir = Path(str(cfg["data_dir"]))
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory does not exist: {dataset_dir}")
    meta_path = dataset_dir / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"Dataset meta.json does not exist: {meta_path}")
    meta = load_json(meta_path)

    prompt_bank_dir = Path(str(cfg["prompt_bank_dir"]))
    meta_prompt_bank_dir = meta.get("prompt_bank_dir")
    if not cfg.get("prompt_bank_dir_explicit") and meta_prompt_bank_dir:
        prompt_bank_dir = Path(str(meta_prompt_bank_dir))
    if not prompt_bank_dir.exists():
        raise SystemExit(f"Prompt bank directory does not exist: {prompt_bank_dir}")

    eta = float(meta.get("eta", cfg["eta"])) if meta.get("eta", cfg["eta"]) is not None else None

    train_y = torch.load(dataset_dir / "train_y.pt", map_location="cpu")
    clean_targets = torch.load(dataset_dir / "clean_train_cot_ids.pt", map_location="cpu")
    clean_train_prompt_ids = torch.load(dataset_dir / "clean_train_prompt_ids.pt", map_location="cpu")
    prompt_len = int(meta.get("prompt_len", clean_train_prompt_ids.size(1)))
    target_len = int(meta.get("target_len", meta.get("cot_len", clean_targets.size(1))))
    final_answer_len = int(meta.get("final_answer_len", meta.get("answer_len", 7)))
    noisy_targets = target_ids_from_train_y(train_y, prompt_len=prompt_len, target_len=target_len)

    max_rows = cfg.get("max_rows")
    if max_rows is not None and int(max_rows) > 0:
        max_rows = min(int(max_rows), int(noisy_targets.size(0)))
        noisy_targets = noisy_targets[:max_rows]
        clean_targets = clean_targets[:max_rows]
        clean_train_prompt_ids = clean_train_prompt_ids[:max_rows]

    metrics = compute_s5_offline_audit_metrics(
        noisy_targets,
        clean_targets,
        eta=eta,
        final_answer_len=final_answer_len,
        chunk_size=int(cfg["chunk_size"]),
    )
    teacher_law = str(meta.get("teacher_law", cfg.get("teacher_law", DEFAULT_TEACHER_LAW)))
    semantic_metrics = None
    key_masks = None
    if teacher_law == SEMANTIC_KEY_NOISE_LAW:
        semantic_config = meta.get("semantic_key_noise", cfg.get("semantic_key_noise"))
        semantic_metrics = compute_semantic_key_noise_audit_metrics(
            noisy_targets,
            clean_targets,
            clean_train_prompt_ids,
            eta=eta,
            semantic_key_noise_config=semantic_config,
            chunk_size=int(cfg["chunk_size"]),
        )
        key_masks = semantic_key_mask(
            clean_train_prompt_ids,
            target_len,
            semantic_key_noise_config_from_obj(semantic_config),
        )
    examples: list[dict[str, Any]] = []
    if not cfg.get("no_examples"):
        examples = build_examples(
            clean_train_prompt_ids=clean_train_prompt_ids,
            clean_targets=clean_targets,
            noisy_targets=noisy_targets,
            final_answer_len=final_answer_len,
            num_examples=int(cfg["num_examples"]),
            start_row=int(cfg["examples_start"]),
            key_masks=key_masks,
        )

    validation_consistency = clean_validation_consistency_report(
        dataset_dir=dataset_dir,
        prompt_bank_dir=prompt_bank_dir,
        meta=meta,
        prompt_len=prompt_len,
        target_len=target_len,
        final_answer_len=final_answer_len,
    )

    report = {
        "eta": eta,
        "dataset_dir": str(dataset_dir),
        "prompt_bank_dir": str(prompt_bank_dir),
        "dataset_meta": meta,
        "seeds": {
            "bank_seed": cfg.get("bank_seed"),
            "teacher_seed": cfg.get("teacher_seed"),
            "render_seed": meta.get("seed", cfg.get("render_seed")),
        },
        "target_lengths": {
            "prompt_len": prompt_len,
            "target_len": target_len,
            "cot_len": int(meta.get("cot_len", clean_targets.size(1))),
            "final_answer_len": final_answer_len,
        },
        "vocab_size": VOCAB_SIZE,
        "s5_num_possible_final_answers": 120,
        "metrics": metrics,
        "semantic_key_noise_metrics": semantic_metrics,
        "rollout_corruption_check": renderer_rollin_static_check(),
        "validation_consistency": validation_consistency,
        "examples": examples,
        "metric_notes": {
            "empirical_corruption_rate_definition": (
                "value-level mismatch rate against the clean oracle target. The renderer "
                "does not save the Bernoulli corruption mask, and replacement is sampled "
                "uniformly from all S5 digit IDs including the original digit."
            ),
            "corruptible_positions_definition": (
                "positions whose clean target token is one of S5 digits 1..5"
            ),
        },
    }
    return report


def print_summary(report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    validation = report["validation_consistency"]
    rollin = report["rollout_corruption_check"]
    print("S5 noisy offline dataset audit")
    print(f"  dataset_dir: {report['dataset_dir']}")
    print(f"  prompt_bank_dir: {report['prompt_bank_dir']}")
    print(f"  eta: {report['eta']}")
    print(f"  num_examples: {metrics['num_examples']}")
    print(f"  target_len: {metrics['target_len']}")
    print(f"  final_answer_len: {metrics['final_answer_len']}")
    print()
    print("Metrics")
    print(f"  random_final_exact_baseline: {metrics['random_final_exact_baseline']:.8f}")
    print(f"  observed_noisy_final_exact: {metrics['noisy_final_exact']:.8f}")
    print(f"  observed/random: {metrics['observed_over_random_final_exact']:.4f}")
    print(f"  noisy_full_exact: {metrics['noisy_full_exact']:.8f}")
    print(f"  token_match_rate: {metrics['token_match_rate']:.8f}")
    print(f"  corruptible_token_match_rate: {metrics['corruptible_token_match_rate']:.8f}")
    print(f"  noncorruptible_token_match_rate: {metrics['noncorruptible_token_match_rate']:.8f}")
    print(f"  empirical_corruption_rate_all: {metrics['empirical_corruption_rate_all']:.8f}")
    print(f"  empirical_corruption_rate_corruptible: {metrics['empirical_corruption_rate_corruptible']:.8f}")
    expected_value_change = metrics["expected_value_change_rate_corruptible_uniform_including_original"]
    if expected_value_change is not None:
        print(f"  expected_value_change_rate_corruptible_uniform_including_original: {expected_value_change:.8f}")
    print(f"  fraction_no_corruption: {metrics['fraction_no_corruption']:.8f}")
    print(f"  fraction_with_corrupted_final_answer_tokens: {metrics['fraction_with_corrupted_final_answer_tokens']:.8f}")
    print(
        "  fraction_with_clean_final_answer_despite_earlier_cot_corruption: "
        f"{metrics['fraction_with_clean_final_answer_despite_earlier_cot_corruption']:.8f}"
    )
    semantic = report.get("semantic_key_noise_metrics")
    if semantic is not None:
        print()
        print("Semantic key noise")
        key_counts = semantic["key_positions_per_trajectory"]
        print(
            "  key_positions_per_trajectory: "
            f"mean={key_counts['mean']:.4f}, min={key_counts['min']}, max={key_counts['max']}"
        )
        print(f"  semantic_key_token_mismatch_rate: {semantic['semantic_key_token_mismatch_rate']:.8f}")
        print(
            "  empirical_non_key_direct_corruption_or_drift_rate: "
            f"{semantic['empirical_non_key_direct_corruption_or_drift_rate']:.8f}"
        )
        print(
            "  fraction_all_key_tokens_matching_clean: "
            f"{semantic['fraction_all_key_tokens_matching_clean']:.8f}"
        )
        baseline = semantic["expected_all_key_clean_rate_uniform_including_clean"]
        if baseline is not None:
            print(f"  expected_all_key_clean_rate_uniform_including_clean: {baseline:.8f}")
    print()
    print("Roll-in check")
    print(f"  feeds_next_query_from_post_corruption_token: {rollin['feeds_next_query_from_post_corruption_token']}")
    print(
        "  feeds_next_query_from_sampled_probability_rollout_token: "
        f"{rollin['feeds_next_query_from_sampled_probability_rollout_token']}"
    )
    print(f"  evidence: {rollin['evidence']}")
    print()
    print("Validation consistency")
    print(f"  offline_bc_validation_target_source: {validation['offline_bc_validation_target_source']}")
    print(f"  opd_nail_validation_target_source: {validation['opd_nail_validation_target_source']}")
    print(f"  final_answer_tokens_in_validation_target: {validation['checks'][-1]['ok']}")
    for check in validation["checks"]:
        print(f"  {check['name']}: {check['ok']}")


def print_examples(examples: list[dict[str, Any]]) -> None:
    if not examples:
        return
    print()
    print("Decoded examples")
    for ex in examples:
        print()
        print(f"Example row {ex['row']}")
        print(f"  prompt: {ex['prompt']}")
        print(f"  clean_final_answer: {ex['clean_final_answer']}")
        print(f"  noisy_final_answer: {ex['noisy_final_answer']}")
        print(f"  final_answer_exact: {ex['final_answer_exact']}")
        print(f"  clean_target: {ex['clean_target']}")
        print(f"  noisy_target: {ex['noisy_target']}")
        print(f"  num_differences: {ex['num_differences']}")
        if ex["semantic_key_positions"]:
            print(f"  semantic_key_positions: {ex['semantic_key_positions']}")
            print(f"  semantic_key_difference_positions: {ex['semantic_key_difference_positions']}")
            print(
                "  first_semantic_key_corruption_position: "
                f"{ex['first_semantic_key_corruption_position']}"
            )
        if ex["differences"]:
            print("  differences:")
            for diff in ex["differences"]:
                print(
                    "    "
                    f"{diff['position']}: {diff['clean_token']}->{diff['noisy_token']} "
                    f"corruptible={diff['corruptible']} "
                    f"in_final_answer={diff['in_final_answer']}"
                )
        suffix = ex["rollout_suffix_after_first_difference"]
        if suffix is not None:
            print(
                "  suffix_after_first_difference "
                f"(pos {suffix['first_difference_position']}):"
            )
            print(f"    clean: {suffix['clean_suffix_from_first_difference']}")
            print(f"    noisy: {suffix['noisy_suffix_from_first_difference']}")


def output_paths(report: dict[str, Any], cfg: dict[str, Any]) -> tuple[Path | None, Path | None]:
    json_path = Path(str(cfg["json_out"])) if cfg.get("json_out") else None
    csv_path = Path(str(cfg["csv_out"])) if cfg.get("csv_out") else None
    output_dir = cfg.get("output_dir")
    if json_path is None and output_dir and not cfg.get("no_json"):
        eta_tag = _float_tag(report["eta"])
        json_path = Path(str(output_dir)) / f"audit_s5_offline_eta{eta_tag}.json"
    return json_path, csv_path


def main() -> None:
    args = parse_args()
    cfg = resolved_config(args)
    report = audit_dataset(cfg)
    print_summary(report)
    print_examples(report["examples"])

    json_path, csv_path = output_paths(report, cfg)
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print()
        print(f"Saved JSON report: {json_path}")
    if csv_path is not None:
        save_csv_row(csv_path, flatten_metrics_for_csv(report))
        print(f"Appended CSV row: {csv_path}")


if __name__ == "__main__":
    main()
