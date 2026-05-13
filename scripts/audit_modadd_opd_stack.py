from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
for path in (ROOT, SRC_ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from data.modular_addition.task import (  # noqa: E402
    corruptible_token_ids as modadd_corruptible_token_ids,
    equals_token_id as modadd_equals_token_id,
    evaluate_saved_clean_modadd_metrics,
)
from nanogpt.methods.student_prefix import (  # noqa: E402
    cached_teacher_token_probs,
    extract_answer_logits,
    forward_kl_simple_loss,
    normalize_student_prefix_method,
    reverse_kl_tm_loss,
    rollout_student,
    sample_teacher_actions,
)
from data.synthetic.eval import teacher_forced_exact_batch  # noqa: E402
from data.synthetic.prompt_bank import load_prompt_bank, select_train_subset  # noqa: E402
from data.synthetic.random_suffix_noise import RANDOM_SUFFIX_AFTER_ERROR_LAW  # noqa: E402
from nanogpt_checkpoint import load_nanogpt_model  # noqa: E402


NOISY_BC_RE = re.compile(
    r"^out-modadd-noisy-bc-"
    r"p(?P<p>\d+)-m(?P<m>\d+)-n(?P<subset_size>\d+)-eta(?P<eta>[^-]+)"
    r"(?:-(?P<rollout_mode>[^-]+))?-seed(?P<seed>\d+)$"
)
OPD_RE = re.compile(
    r"^out-modadd-opd-(?P<objective>.+?)-"
    r"p(?P<p>\d+)-m(?P<m>\d+)-n(?P<subset_size>\d+)-eta(?P<eta>[^-]+)-"
    r"(?P<teacher_law>[^-]+)-(?P<temp_tag>[^-]+)-seed(?P<seed>\d+)$"
)
STUDENT_PREFIX_RE = re.compile(
    r"^out-modadd-(?P<method_family>opd|nail)-(?P<loss>forward|reverse)-(?P<teacher_signal>mc|full)-"
    r"p(?P<p>\d+)-m(?P<m>\d+)-n(?P<subset_size>\d+)-eta(?P<eta>[^-]+)-"
    r"(?P<teacher_law>[^-]+)(?P<temp_suffix>(?:-.*)?)\-seed(?P<seed>\d+)$"
)

EXPECTED_METHODS = ("LogLossBC", "NAIL-F", "NAIL-R", "OPD-R")
DEFAULT_ETAS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9)


@dataclass
class RunRecord:
    run_id: str
    method: str
    objective: str
    eta: float
    seed: int
    out_dir: Path
    teacher_law: str
    rollout_mode: str
    temp_tag: str
    launcher_config: dict[str, Any]
    run_meta: dict[str, Any] | None
    last_eval: dict[str, Any] | None
    dataset_dir: Path | None
    dataset_meta: dict[str, Any] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the ModAdd LogLossBC, NAIL-F/R, and OPD-F/R stack for one p,m,subset,seed sweep."
        )
    )
    parser.add_argument("--root", type=Path, default=ROOT, help="Repo root to scan.")
    parser.add_argument("--p", type=int, default=7)
    parser.add_argument("--m", type=int, default=127)
    parser.add_argument("--subset-size", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20260417)
    parser.add_argument(
        "--etas",
        type=float,
        nargs="*",
        default=list(DEFAULT_ETAS),
        help="Eta values to include.",
    )
    parser.add_argument(
        "--diagnostic-batch-size",
        type=int,
        default=512,
        help="Number of training prompts to use for objective diagnostics.",
    )
    parser.add_argument(
        "--eval-n",
        type=int,
        default=256,
        help="Number of clean validation prompts for greedy/sample eval. Use <=0 for full val.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument(
        "--sample-eval-temperature",
        type=float,
        default=1.0,
        help="Temperature for sampled clean evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device to use. Defaults to cuda if available, else cpu.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "debugging-log" / "modadd_p7_m127_opd_audit",
        help="Prefix for the summary JSON and CSV outputs.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with an error if any expected run is missing.",
    )
    return parser.parse_args()


def parse_eta_tag(tag: str) -> float:
    return float(tag.replace("p", ".").replace("neg", "-"))


def float_tag(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text.replace(".", "p").replace("-", "neg")


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_last_eval(out_dir: Path) -> dict[str, Any] | None:
    path = out_dir / "last_eval.json"
    if not path.exists():
        return None
    return load_json(path)


def normalize_repo_path(value: str | None, root: Path) -> str | None:
    if value is None or value == "":
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    try:
        return str(path.resolve())
    except FileNotFoundError:
        return str(path)


def extract_method_from_objective(objective: str) -> str | None:
    if objective == "forward_kl_simple":
        return "NAIL-F"
    if objective == "reverse_kl_tm":
        return "OPD-R"
    if objective.startswith("reverse_kl_"):
        return "NAIL-R"
    return None


def method_from_student_prefix_state(run_meta: dict[str, Any]) -> str:
    state = normalize_student_prefix_method(run_meta)
    if state["method_family"] == "opd":
        return "OPD-R"
    if state["loss"] == "forward":
        return "NAIL-F"
    return "NAIL-R"


def objective_from_student_prefix_state(run_meta: dict[str, Any]) -> str:
    state = normalize_student_prefix_method(run_meta)
    if state["teacher_signal"] == "mc" and state["loss"] == "forward":
        return "forward_kl_simple"
    if state["teacher_signal"] == "full" and state["loss"] == "forward":
        return "forward_kl_full"
    if state["teacher_signal"] == "mc" and state["method_family"] == "opd":
        return "reverse_kl_tm"
    if state["teacher_signal"] == "mc":
        return "reverse_kl_simple"
    return "reverse_kl_full"


def normalize_rollout_mode_tag(rollout_mode: str | None) -> str:
    if rollout_mode in (None, "", "greedy", "greedy_then_corrupt"):
        return "greedy_then_corrupt"
    if rollout_mode in ("sample", "sample_then_corrupt"):
        return "sample_then_corrupt"
    return str(rollout_mode)


def expected_teacher_law_for_rollout_mode(rollout_mode: str) -> str:
    rollout_mode = normalize_rollout_mode_tag(rollout_mode)
    if rollout_mode == "sample_then_corrupt":
        return "distributional_noise"
    if rollout_mode == "greedy_then_corrupt":
        return "corrupted_greedy"
    return "unknown"


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def stats_dict(values: torch.Tensor) -> dict[str, float | None]:
    work = values.detach().float().reshape(-1).cpu()
    if work.numel() == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "mean": safe_float(work.mean().item()),
        "std": safe_float(work.std(unbiased=False).item()),
        "min": safe_float(work.min().item()),
        "max": safe_float(work.max().item()),
    }


def per_step_stats(values: torch.Tensor) -> list[dict[str, float | None]]:
    work = values.detach().float().cpu()
    return [
        {
            "step": int(step),
            **stats_dict(work[:, step]),
        }
        for step in range(work.size(1))
    ]


def prompt_bank_dir_for_run(run: RunRecord, root: Path) -> Path | None:
    if run.method == "LogLossBC":
        value = run.dataset_meta.get("prompt_bank_dir") if run.dataset_meta is not None else None
    else:
        value = run.run_meta.get("prompt_bank_dir") if run.run_meta is not None else run.launcher_config.get("prompt_bank_dir")
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path


def teacher_checkpoint_for_run(run: RunRecord, root: Path) -> Path | None:
    if run.method == "LogLossBC":
        value = run.dataset_meta.get("teacher_checkpoint") if run.dataset_meta is not None else None
    else:
        value = run.run_meta.get("teacher_checkpoint") if run.run_meta is not None else run.launcher_config.get("teacher_checkpoint")
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path


def load_subset_indices(path: Path) -> torch.Tensor | None:
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu").to(dtype=torch.long)


def discover_runs(
    root: Path,
    *,
    p: int,
    m: int,
    subset_size: int,
    seed: int,
    etas: list[float],
) -> list[RunRecord]:
    eta_set = {float(eta) for eta in etas}
    records: list[RunRecord] = []

    for out_dir in sorted(root.rglob("out-modadd-*")):
        if not out_dir.is_dir():
            continue
        if not (out_dir / "eval_history.jsonl").exists():
            continue
        name = out_dir.name
        launcher_config_path = out_dir / "launcher_config.json"
        launcher_config = load_json(launcher_config_path) if launcher_config_path.exists() else {}
        last_eval = load_last_eval(out_dir)

        noisy_match = NOISY_BC_RE.match(name)
        if noisy_match:
            run_p = int(noisy_match.group("p"))
            run_m = int(noisy_match.group("m"))
            run_n = int(noisy_match.group("subset_size"))
            run_seed = int(noisy_match.group("seed"))
            run_eta = parse_eta_tag(noisy_match.group("eta"))
            if (run_p, run_m, run_n, run_seed) != (p, m, subset_size, seed):
                continue
            if run_eta not in eta_set:
                continue
            dataset_name = launcher_config.get("dataset")
            dataset_dir = root / "data" / dataset_name if dataset_name else None
            dataset_meta = load_json(dataset_dir / "meta.json") if dataset_dir is not None and (dataset_dir / "meta.json").exists() else None
            records.append(
                RunRecord(
                    run_id=name,
                    method="LogLossBC",
                    objective="sample_then_corrupt",
                    eta=run_eta,
                    seed=run_seed,
                    out_dir=out_dir,
                    teacher_law=expected_teacher_law_for_rollout_mode(
                        normalize_rollout_mode_tag(noisy_match.group("rollout_mode"))
                    ),
                    rollout_mode=normalize_rollout_mode_tag(noisy_match.group("rollout_mode")),
                    temp_tag="",
                    launcher_config=launcher_config,
                    run_meta=None,
                    last_eval=last_eval,
                    dataset_dir=dataset_dir,
                    dataset_meta=dataset_meta,
                )
            )
            continue

        run_meta_path = out_dir / "run_meta.json"
        run_meta = load_json(run_meta_path) if run_meta_path.exists() else None

        student_prefix_match = STUDENT_PREFIX_RE.match(name)
        if student_prefix_match:
            run_p = int(student_prefix_match.group("p"))
            run_m = int(student_prefix_match.group("m"))
            run_n = int(student_prefix_match.group("subset_size"))
            run_seed = int(student_prefix_match.group("seed"))
            run_eta = parse_eta_tag(student_prefix_match.group("eta"))
            if (run_p, run_m, run_n, run_seed) != (p, m, subset_size, seed):
                continue
            if run_eta not in eta_set:
                continue
            if run_meta is not None:
                method = method_from_student_prefix_state(run_meta)
                objective = objective_from_student_prefix_state(run_meta)
            else:
                method = "OPD-R" if student_prefix_match.group("method_family") == "opd" else (
                    "NAIL-F" if student_prefix_match.group("loss") == "forward" else "NAIL-R"
                )
                objective = (
                    f"{student_prefix_match.group('method_family')}:"
                    f"{student_prefix_match.group('loss')}:"
                    f"{student_prefix_match.group('teacher_signal')}"
                )
            teacher_law = student_prefix_match.group("teacher_law")
            temp_tag = student_prefix_match.group("temp_suffix").lstrip("-")
        else:
            opd_match = OPD_RE.match(name)
            if not opd_match:
                continue
            objective = opd_match.group("objective")
            run_p = int(opd_match.group("p"))
            run_m = int(opd_match.group("m"))
            run_n = int(opd_match.group("subset_size"))
            run_seed = int(opd_match.group("seed"))
            run_eta = parse_eta_tag(opd_match.group("eta"))
            if (run_p, run_m, run_n, run_seed) != (p, m, subset_size, seed):
                continue
            if run_eta not in eta_set:
                continue
            if run_meta is not None and ("method_family" in run_meta or "objective" in run_meta):
                method = method_from_student_prefix_state(run_meta)
                objective = objective_from_student_prefix_state(run_meta)
            else:
                method = extract_method_from_objective(objective)
            if method is None:
                continue
            teacher_law = opd_match.group("teacher_law")
            temp_tag = opd_match.group("temp_tag")

        records.append(
            RunRecord(
                run_id=name,
                method=method,
                objective=objective,
                eta=run_eta,
                seed=run_seed,
                out_dir=out_dir,
                teacher_law=teacher_law,
                rollout_mode="",
                temp_tag=temp_tag,
                launcher_config=launcher_config,
                run_meta=run_meta,
                last_eval=last_eval,
                dataset_dir=None,
                dataset_meta=None,
            )
        )

    return records


def expected_run_status(records: list[RunRecord], etas: list[float]) -> dict[str, Any]:
    by_eta_method: dict[tuple[float, str], list[str]] = defaultdict(list)
    for record in records:
        by_eta_method[(record.eta, record.method)].append(record.run_id)

    coverage_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    duplicates: list[str] = []
    for eta in sorted(etas):
        for method in EXPECTED_METHODS:
            found = by_eta_method.get((eta, method), [])
            coverage_rows.append(
                {
                    "eta": eta,
                    "method": method,
                    "count": len(found),
                    "run_ids": found,
                }
            )
            if len(found) == 0:
                missing.append(f"eta={eta}: missing {method}")
            if len(found) > 1:
                duplicates.append(f"eta={eta}: duplicate {method} runs {found}")
    return {
        "coverage_rows": coverage_rows,
        "missing_expected_runs": missing,
        "duplicate_runs": duplicates,
    }


def load_model_cached(
    checkpoint_or_dir: Path,
    *,
    device: str,
    cache: dict[str, torch.nn.Module],
) -> torch.nn.Module:
    key = str(checkpoint_or_dir.resolve())
    if key not in cache:
        cache[key] = load_nanogpt_model(
            checkpoint_or_dir,
            map_location="cpu",
            device=device,
            eval_mode=True,
        )
    return cache[key]


@torch.no_grad()
def sampled_generate_target_ids_batched(
    model,
    prompt_ids_batch: torch.Tensor,
    *,
    max_new_tokens: int,
    device: str,
    temperature: float,
) -> torch.Tensor:
    prompt = prompt_ids_batch.to(device=device, dtype=torch.long)
    generated = torch.empty((prompt.size(0), max_new_tokens), dtype=torch.long, device=device)
    input_ids = prompt
    past_key_values = None

    for step in range(max_new_tokens):
        logits, _, past_key_values = model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        next_logits = logits[:, -1, :].float()
        if temperature > 0:
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_id = torch.argmax(next_logits, dim=-1)
        generated[:, step] = next_id
        input_ids = next_id.unsqueeze(1)

    return generated.to(device="cpu", dtype=torch.long)


@torch.no_grad()
def evaluate_saved_clean_modadd_metrics_sampled(
    model,
    *,
    device: str,
    data_dir: Path,
    n_eval: int | None,
    batch_size: int,
    temperature: float,
) -> dict[str, float]:
    prompt_ids_all = torch.load(data_dir / "clean_val_prompt_ids.pt", map_location="cpu").long()
    target_ids_all = torch.load(data_dir / "clean_val_cot_ids.pt", map_location="cpu").long()
    if n_eval is not None:
        prompt_ids_all = prompt_ids_all[:n_eval]
        target_ids_all = target_ids_all[:n_eval]

    teacher_forced_ok = 0
    sampled_full_ok = 0
    sampled_final_ok = 0
    n = prompt_ids_all.size(0)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        prompt_ids_batch = prompt_ids_all[start:end]
        target_ids_batch = target_ids_all[start:end]
        tf_ok = teacher_forced_exact_batch(model, prompt_ids_batch, target_ids_batch, device)
        teacher_forced_ok += int(tf_ok.sum().item())

        pred_ids_batch = sampled_generate_target_ids_batched(
            model,
            prompt_ids_batch,
            max_new_tokens=target_ids_batch.size(1),
            device=device,
            temperature=temperature,
        )
        sampled_full_ok += int(pred_ids_batch.eq(target_ids_batch).all(dim=1).sum().item())
        sampled_final_ok += int(pred_ids_batch[:, -1:].eq(target_ids_batch[:, -1:]).all(dim=1).sum().item())

    return {
        "cot_exact": teacher_forced_ok / n,
        "clean_full_exact": sampled_full_ok / n,
        "clean_final_exact": sampled_final_ok / n,
    }


def reconstruct_offline_targets(dataset_dir: Path, prompt_len: int, limit: int) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_ids = torch.load(dataset_dir / "clean_train_prompt_ids.pt", map_location="cpu").long()[:limit]
    train_y = torch.load(dataset_dir / "train_y.pt", map_location="cpu").long()[:limit]
    target_ids = train_y[:, prompt_len - 1:].contiguous()
    return prompt_ids, target_ids


def compare_offline_targets_to_teacher(
    *,
    run: RunRecord,
    teacher,
    prompt_len: int,
    vocab_size: int,
    limit: int,
    device: str,
    p: int,
) -> dict[str, Any]:
    assert run.dataset_dir is not None
    prompt_ids, noisy_targets = reconstruct_offline_targets(run.dataset_dir, prompt_len, limit)
    teacher_probs = cached_teacher_token_probs(
        teacher,
        prompt_ids,
        noisy_targets,
        eta=run.eta,
        teacher_law=expected_teacher_law_for_rollout_mode(run.rollout_mode),
        corruptible_token_ids=tuple(modadd_corruptible_token_ids(p)),
        device=device,
        autocast_context=nullcontext(),
    ).cpu()
    empirical = F.one_hot(noisy_targets, num_classes=vocab_size).float().mean(dim=0)
    expected = teacher_probs.mean(dim=0)
    abs_diff = (empirical - expected).abs()
    per_step = []
    for step in range(abs_diff.size(0)):
        per_step.append(
            {
                "step": int(step),
                "tv_distance": safe_float(0.5 * abs_diff[step].sum().item()),
                "max_abs_diff": safe_float(abs_diff[step].max().item()),
            }
        )
    return {
        "teacher_law_checked": expected_teacher_law_for_rollout_mode(run.rollout_mode),
        "limit": int(limit),
        "overall_mean_tv_distance": safe_float(0.5 * abs_diff.sum(dim=-1).mean().item()),
        "overall_max_abs_diff": safe_float(abs_diff.max().item()),
        "per_step": per_step,
    }


def summarize_teacher_probs(teacher_probs: torch.Tensor, *, corruptible_ids: tuple[int, ...]) -> dict[str, Any]:
    teacher_probs = teacher_probs.detach().float()
    log_teacher_probs = torch.log(teacher_probs.clamp_min(1e-10))
    entropy = -(teacher_probs * log_teacher_probs).sum(dim=-1)
    top1_mass = teacher_probs.max(dim=-1).values
    corruptible_mass = teacher_probs.index_select(
        dim=-1,
        index=torch.as_tensor(corruptible_ids, dtype=torch.long, device=teacher_probs.device),
    ).sum(dim=-1)
    return {
        "overall_entropy": stats_dict(entropy),
        "overall_top1_mass": stats_dict(top1_mass),
        "overall_corruptible_mass": stats_dict(corruptible_mass),
        "per_step_entropy": per_step_stats(entropy),
        "per_step_top1_mass": per_step_stats(top1_mass),
        "per_step_corruptible_mass": per_step_stats(corruptible_mass),
    }


def forward_kl_simple_diagnostics(
    *,
    answer_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    policy_temperature: float | None,
) -> dict[str, Any]:
    teacher_targets = sample_teacher_actions(teacher_probs)
    loss, stats = forward_kl_simple_loss(
        answer_logits,
        teacher_targets,
        teacher_probs=teacher_probs,
        temperature=policy_temperature,
        eps=1e-10,
    )
    teacher_target_probs = teacher_probs.gather(2, teacher_targets.unsqueeze(-1)).squeeze(-1)
    target_hist = torch.bincount(
        teacher_targets.reshape(-1).cpu(),
        minlength=teacher_probs.size(-1),
    )
    return {
        "loss": safe_float(loss.item()),
        "target_histogram": [int(value) for value in target_hist.tolist()],
        "overall_teacher_target_prob": stats_dict(teacher_target_probs),
        "overall_log_teacher_target": stats_dict(stats["log_teacher_target"]),
        "overall_log_student_target": stats_dict(stats["log_student_target"]),
        "per_step_teacher_target_prob": per_step_stats(teacher_target_probs),
        "per_step_log_teacher_target": per_step_stats(stats["log_teacher_target"]),
        "per_step_log_student_target": per_step_stats(stats["log_student_target"]),
    }


def reverse_kl_tm_diagnostics(
    *,
    answer_logits: torch.Tensor,
    actions: torch.Tensor,
    log_q: torch.Tensor,
    teacher_probs: torch.Tensor,
) -> dict[str, Any]:
    loss, stats = reverse_kl_tm_loss(
        answer_logits,
        actions,
        log_q=log_q,
        teacher_probs=teacher_probs,
        eps=1e-10,
    )
    weights = stats["importance_weight"].detach().float().reshape(-1).cpu()
    weight_sum = weights.sum().item()
    weight_sq_sum = torch.square(weights).sum().item()
    ess = None
    ess_ratio = None
    if weight_sq_sum > 0:
        ess = weight_sum * weight_sum / weight_sq_sum
        ess_ratio = ess / float(weights.numel())
    return {
        "loss": safe_float(loss.item()),
        "effective_sample_size": safe_float(ess),
        "effective_sample_size_ratio": safe_float(ess_ratio),
        "overall_log_q": stats_dict(log_q),
        "overall_log_teacher": stats_dict(stats["log_teacher"]),
        "overall_advantage": stats_dict(stats["advantage"]),
        "overall_importance_weight": stats_dict(stats["importance_weight"]),
        "per_step_log_q": per_step_stats(log_q),
        "per_step_log_teacher": per_step_stats(stats["log_teacher"]),
        "per_step_advantage": per_step_stats(stats["advantage"]),
        "per_step_importance_weight": per_step_stats(stats["importance_weight"]),
    }


def objective_diagnostics_for_run(
    *,
    run: RunRecord,
    prompt_batch: torch.Tensor,
    clean_target_batch: torch.Tensor,
    prompt_bank,
    teacher,
    student,
    device: str,
    p: int,
) -> dict[str, Any]:
    policy_temperature = None
    if run.objective == "forward_kl_simple":
        if run.run_meta is not None:
            policy_temperature = float(
                run.run_meta.get(
                    "resolved_loss_temperature",
                    run.run_meta.get("student_temperature", 1.0),
                )
            )
        else:
            policy_temperature = 1.0

    rollout_temperature = 0.0
    if run.run_meta is not None:
        rollout_temperature = float(
            run.run_meta.get(
                "resolved_rollout_temperature",
                run.run_meta.get("student_rollout_temperature", run.run_meta.get("student_temperature", 0.0)),
            )
        )

    full_seq, actions, log_q = rollout_student(
        student,
        prompt_batch,
        target_len=prompt_bank.cot_len,
        temperature=rollout_temperature,
        device=device,
        autocast_context=nullcontext(),
    )
    teacher_prob_kwargs: dict[str, Any] = {}
    if run.teacher_law == RANDOM_SUFFIX_AFTER_ERROR_LAW:
        teacher_prob_kwargs = {
            "clean_target_ids": clean_target_batch,
            "random_suffix_noise_config": (run.run_meta or {}).get("random_suffix_noise"),
            "task_name": "modadd",
        }
    teacher_probs = cached_teacher_token_probs(
        teacher,
        prompt_batch,
        actions,
        eta=run.eta,
        teacher_law=run.teacher_law,
        corruptible_token_ids=tuple(modadd_corruptible_token_ids(p)),
        device=device,
        autocast_context=nullcontext(),
        **teacher_prob_kwargs,
    )
    rollout_inputs = full_seq[:, :-1]
    logits, _ = student(rollout_inputs.to(device=device, dtype=torch.long), return_full_logits=True)
    answer_logits = extract_answer_logits(
        logits,
        prompt_len=prompt_bank.prompt_len,
        target_len=prompt_bank.cot_len,
    )
    teacher_summary = summarize_teacher_probs(
        teacher_probs,
        corruptible_ids=tuple(modadd_corruptible_token_ids(p)),
    )
    if run.objective == "forward_kl_simple":
        objective_summary = forward_kl_simple_diagnostics(
            answer_logits=answer_logits,
            teacher_probs=teacher_probs,
            policy_temperature=policy_temperature,
        )
    elif run.objective == "reverse_kl_tm":
        objective_summary = reverse_kl_tm_diagnostics(
            answer_logits=answer_logits,
            actions=actions,
            log_q=log_q,
            teacher_probs=teacher_probs,
        )
    else:
        objective_summary = {"note": f"Objective {run.objective} has no specialized audit block."}
    return {
        "rollout_action_histogram": [
            int(value)
            for value in torch.bincount(actions.reshape(-1).cpu(), minlength=teacher_probs.size(-1)).tolist()
        ],
        "teacher_distribution_summary": teacher_summary,
        "objective_summary": objective_summary,
    }


def eval_metrics_for_run(
    *,
    run: RunRecord,
    model,
    prompt_bank_dir: Path,
    eval_n: int | None,
    eval_batch_size: int,
    sample_eval_temperature: float,
    device: str,
) -> dict[str, Any]:
    greedy = evaluate_saved_clean_modadd_metrics(
        model,
        device=device,
        data_dir=prompt_bank_dir,
        n_eval=eval_n,
        batch_size=eval_batch_size,
    )
    sampled = evaluate_saved_clean_modadd_metrics_sampled(
        model,
        device=device,
        data_dir=prompt_bank_dir,
        n_eval=eval_n,
        batch_size=eval_batch_size,
        temperature=sample_eval_temperature,
    )
    return {
        "logged_last_eval": {
            "val/clean_full_exact": safe_float(
                (run.last_eval or {}).get("val/clean_full_exact", (run.last_eval or {}).get("val_clean_full_exact"))
            ),
            "val/clean_final_exact": safe_float(
                (run.last_eval or {}).get("val/clean_final_exact", (run.last_eval or {}).get("val_clean_final_exact"))
            ),
        },
        "greedy_eval": {
            "clean_full_exact": safe_float(greedy["clean_full_exact"]),
            "clean_final_exact": safe_float(greedy["clean_final_exact"]),
            "cot_exact": safe_float(greedy["cot_exact"]),
        },
        "sampled_eval": {
            "temperature": safe_float(sample_eval_temperature),
            "clean_full_exact": safe_float(sampled["clean_full_exact"]),
            "clean_final_exact": safe_float(sampled["clean_final_exact"]),
            "cot_exact": safe_float(sampled["cot_exact"]),
        },
    }


def subset_match_summary(
    *,
    run: RunRecord,
    expected_subset: torch.Tensor,
) -> dict[str, Any]:
    if run.method == "LogLossBC":
        subset_path = run.dataset_dir / "subset_indices.pt" if run.dataset_dir is not None else None
    else:
        subset_path = run.out_dir / "subset_indices.pt"
    if subset_path is None or not subset_path.exists():
        return {"subset_indices_saved": False, "matches_expected_prefix": None}
    subset_indices = load_subset_indices(subset_path)
    return {
        "subset_indices_saved": True,
        "matches_expected_prefix": bool(torch.equal(subset_indices, expected_subset)),
        "num_indices": int(subset_indices.numel()),
    }


def parity_checks(
    *,
    records: list[RunRecord],
    root: Path,
    p: int,
    subset_size: int,
) -> dict[str, Any]:
    values_by_key: dict[str, set[str]] = defaultdict(set)
    subset_checks: dict[str, Any] = {}

    prompt_bank_paths: dict[str, Any] = {}
    prompt_bank_cache: dict[str, Any] = {}
    expected_subsets: dict[str, torch.Tensor] = {}

    for run in records:
        prompt_bank_dir = prompt_bank_dir_for_run(run, root)
        teacher_checkpoint = teacher_checkpoint_for_run(run, root)
        prompt_bank_key = normalize_repo_path(str(prompt_bank_dir), root) if prompt_bank_dir is not None else None
        teacher_key = normalize_repo_path(str(teacher_checkpoint), root) if teacher_checkpoint is not None else None
        if prompt_bank_key is not None:
            values_by_key["prompt_bank_dir"].add(prompt_bank_key)
            prompt_bank_paths[run.run_id] = prompt_bank_key
        if teacher_key is not None:
            values_by_key["teacher_checkpoint"].add(teacher_key)
        values_by_key["subset_size"].add(str(subset_size))
        values_by_key["train_seed"].add(str(run.seed))
        if run.method == "LogLossBC":
            values_by_key["offline_rollout_mode"].add(run.rollout_mode)
            values_by_key["offline_target_type"].add(
                str((run.launcher_config or {}).get("offline_target_type", "tokens"))
            )
            values_by_key["offline_expected_teacher_law"].add(
                expected_teacher_law_for_rollout_mode(run.rollout_mode)
            )
        else:
            values_by_key["online_teacher_law"].add(run.teacher_law)
            values_by_key["online_objective"].add(run.objective)
            if run.run_meta is not None:
                values_by_key["student_temperature"].add(str(run.run_meta.get("student_temperature")))

        if prompt_bank_key is not None and prompt_bank_key not in prompt_bank_cache:
            prompt_bank_cache[prompt_bank_key] = load_prompt_bank(prompt_bank_dir)
            expected_subsets[prompt_bank_key] = select_train_subset(prompt_bank_cache[prompt_bank_key], subset_size)
        if prompt_bank_key is not None:
            subset_checks[run.run_id] = subset_match_summary(
                run=run,
                expected_subset=expected_subsets[prompt_bank_key],
            )

    equivalence_rows: list[dict[str, Any]] = []
    for run in records:
        if run.method != "LogLossBC":
            continue
        expected_law = expected_teacher_law_for_rollout_mode(run.rollout_mode)
        equivalence_rows.append(
            {
                "eta": run.eta,
                "offline_rollout_mode": run.rollout_mode,
                "matched_online_teacher_law": expected_law,
                "is_distributional_match": expected_law == "distributional_noise",
            }
        )

    return {
        "shared_values": {key: sorted(values) for key, values in sorted(values_by_key.items())},
        "subset_match": subset_checks,
        "teacher_side_equivalence_rows": equivalence_rows,
        "corruption_set": {
            "corruptible_token_ids": list(modadd_corruptible_token_ids(p)),
            "equals_token_id": int(modadd_equals_token_id(p)),
            "equals_token_excluded_from_corruption": int(modadd_equals_token_id(p)) not in set(modadd_corruptible_token_ids(p)),
        },
    }


def method_gap_summary(eval_summary: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    by_eta: dict[float, dict[str, dict[str, Any]]] = defaultdict(dict)
    for run_id, payload in eval_summary.items():
        eta = payload["eta"]
        by_eta[eta][payload["method"]] = payload

    rows: list[dict[str, Any]] = []
    for eta in sorted(by_eta):
        methods = by_eta[eta]
        offline = methods.get("LogLossBC")
        nail_forward = methods.get("NAIL-F")
        nail_reverse = methods.get("NAIL-R")
        opd = methods.get("OPD-R")
        row: dict[str, Any] = {"eta": eta}
        if offline is not None and nail_forward is not None:
            row["greedy_clean_full_gap_loglossbc_minus_nail_forward"] = safe_float(
                offline["greedy_eval"]["clean_full_exact"] - nail_forward["greedy_eval"]["clean_full_exact"]
            )
            row["sampled_clean_full_gap_loglossbc_minus_nail_forward"] = safe_float(
                offline["sampled_eval"]["clean_full_exact"] - nail_forward["sampled_eval"]["clean_full_exact"]
            )
            row["greedy_clean_final_gap_loglossbc_minus_nail_forward"] = safe_float(
                offline["greedy_eval"]["clean_final_exact"] - nail_forward["greedy_eval"]["clean_final_exact"]
            )
            row["sampled_clean_final_gap_loglossbc_minus_nail_forward"] = safe_float(
                offline["sampled_eval"]["clean_final_exact"] - nail_forward["sampled_eval"]["clean_final_exact"]
            )
        if nail_forward is not None and opd is not None:
            row["greedy_clean_full_gap_nail_forward_minus_opd"] = safe_float(
                nail_forward["greedy_eval"]["clean_full_exact"] - opd["greedy_eval"]["clean_full_exact"]
            )
            row["sampled_clean_full_gap_nail_forward_minus_opd"] = safe_float(
                nail_forward["sampled_eval"]["clean_full_exact"] - opd["sampled_eval"]["clean_full_exact"]
            )
            row["greedy_clean_final_gap_nail_forward_minus_opd"] = safe_float(
                nail_forward["greedy_eval"]["clean_final_exact"] - opd["greedy_eval"]["clean_final_exact"]
            )
            row["sampled_clean_final_gap_nail_forward_minus_opd"] = safe_float(
                nail_forward["sampled_eval"]["clean_final_exact"] - opd["sampled_eval"]["clean_final_exact"]
            )
        if nail_reverse is not None and opd is not None:
            row["greedy_clean_full_gap_nail_reverse_minus_opd"] = safe_float(
                nail_reverse["greedy_eval"]["clean_full_exact"] - opd["greedy_eval"]["clean_full_exact"]
            )
            row["sampled_clean_full_gap_nail_reverse_minus_opd"] = safe_float(
                nail_reverse["sampled_eval"]["clean_full_exact"] - opd["sampled_eval"]["clean_full_exact"]
            )
        rows.append(row)
    return rows


def rows_for_csv(records: list[RunRecord], root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in sorted(records, key=lambda item: (item.eta, item.method)):
        prompt_bank_dir = prompt_bank_dir_for_run(run, root)
        teacher_checkpoint = teacher_checkpoint_for_run(run, root)
        rows.append(
            {
                "run_id": run.run_id,
                "method": run.method,
                "objective": run.objective,
                "eta": run.eta,
                "seed": run.seed,
                "teacher_law": run.teacher_law,
                "rollout_mode": run.rollout_mode,
                "student_temperature": None if run.run_meta is None else run.run_meta.get("student_temperature"),
                "prompt_bank_dir": None if prompt_bank_dir is None else str(prompt_bank_dir),
                "teacher_checkpoint": None if teacher_checkpoint is None else str(teacher_checkpoint),
                "dataset_dir": None if run.dataset_dir is None else str(run.dataset_dir),
                "logged_final_clean_full_exact": None if run.last_eval is None else run.last_eval.get("val/clean_full_exact", run.last_eval.get("val_clean_full_exact")),
                "logged_final_clean_final_exact": None if run.last_eval is None else run.last_eval.get("val/clean_final_exact", run.last_eval.get("val_clean_final_exact")),
                "out_dir": str(run.out_dir),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["note"])
            writer.writerow(["no rows"])
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    eval_n = None if args.eval_n <= 0 else int(args.eval_n)

    summary: dict[str, Any] = {
        "scope": {
            "root": str(root),
            "p": args.p,
            "m": args.m,
            "subset_size": args.subset_size,
            "seed": args.seed,
            "etas": [float(eta) for eta in args.etas],
            "diagnostic_batch_size": args.diagnostic_batch_size,
            "eval_n": eval_n,
            "eval_batch_size": args.eval_batch_size,
            "sample_eval_temperature": args.sample_eval_temperature,
            "device": device,
        }
    }

    records = discover_runs(
        root,
        p=args.p,
        m=args.m,
        subset_size=args.subset_size,
        seed=args.seed,
        etas=[float(eta) for eta in args.etas],
    )
    summary["discovery"] = expected_run_status(records, [float(eta) for eta in args.etas])
    summary["run_count"] = len(records)
    write_csv(output_prefix.with_name(output_prefix.name + "_runs.csv"), rows_for_csv(records, root))

    if not records:
        summary["status"] = "no_matching_runs_found"
        with open(output_prefix.with_name(output_prefix.name + "_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        if args.strict:
            raise SystemExit("No matching runs found for the requested ModAdd audit scope.")
        print("No matching runs found; wrote discovery-only summary.")
        return

    missing = summary["discovery"]["missing_expected_runs"]
    if missing and args.strict:
        with open(output_prefix.with_name(output_prefix.name + "_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        raise SystemExit(f"Missing expected runs: {missing}")

    summary["parity_checks"] = parity_checks(
        records=records,
        root=root,
        p=args.p,
        subset_size=args.subset_size,
    )

    model_cache: dict[str, torch.nn.Module] = {}
    offline_teacher_checks: dict[str, Any] = {}
    objective_checks: dict[str, Any] = {}
    eval_checks: dict[str, dict[str, Any]] = {}

    for run in sorted(records, key=lambda item: (item.eta, item.method)):
        prompt_bank_dir = prompt_bank_dir_for_run(run, root)
        if prompt_bank_dir is None:
            continue
        prompt_bank = load_prompt_bank(prompt_bank_dir)
        diagnostic_indices = select_train_subset(prompt_bank, args.subset_size)[
            : args.diagnostic_batch_size
        ]
        prompt_batch = prompt_bank.clean_train_prompt_ids.index_select(0, diagnostic_indices)
        clean_target_batch = prompt_bank.clean_train_cot_ids.index_select(
            0,
            diagnostic_indices,
        )
        student = load_model_cached(run.out_dir, device=device, cache=model_cache)
        teacher_checkpoint = teacher_checkpoint_for_run(run, root)
        teacher = None
        if teacher_checkpoint is not None:
            teacher = load_model_cached(teacher_checkpoint, device=device, cache=model_cache)

        if run.method == "LogLossBC" and teacher is not None and run.dataset_dir is not None:
            offline_teacher_checks[run.run_id] = compare_offline_targets_to_teacher(
                run=run,
                teacher=teacher,
                prompt_len=prompt_bank.prompt_len,
                vocab_size=prompt_bank.p + 1,
                limit=args.diagnostic_batch_size,
                device=device,
                p=args.p,
            )

        if run.method != "LogLossBC" and teacher is not None:
            objective_checks[run.run_id] = objective_diagnostics_for_run(
                run=run,
                prompt_batch=prompt_batch,
                clean_target_batch=clean_target_batch,
                prompt_bank=prompt_bank,
                teacher=teacher,
                student=student,
                device=device,
                p=args.p,
            )

        eval_payload = eval_metrics_for_run(
            run=run,
            model=student,
            prompt_bank_dir=prompt_bank_dir,
            eval_n=eval_n,
            eval_batch_size=args.eval_batch_size,
            sample_eval_temperature=args.sample_eval_temperature,
            device=device,
        )
        eval_payload["method"] = run.method
        eval_payload["eta"] = run.eta
        eval_checks[run.run_id] = eval_payload

    summary["teacher_supervision_checks"] = {
        "offline_dataset_vs_analytic_teacher": offline_teacher_checks,
    }
    summary["objective_diagnostics"] = objective_checks
    summary["evaluation_checks"] = eval_checks
    summary["method_gap_summary"] = method_gap_summary(eval_checks)
    summary["status"] = "ok_with_missing_runs" if missing else "ok"

    summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote audit summary to {summary_path}")
    if missing:
        print("Missing expected runs:")
        for item in missing:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
