#!/usr/bin/env bash
set -euo pipefail

# Launch S5 random-suffix online NAIL ablations.
#
# Usage from repo root on the dev node:
#   DRY_RUN=1 scripts/launch_s5_random_suffix_ablations.sh
#   scripts/launch_s5_random_suffix_ablations.sh
#
# Optional knobs:
#   MODE=mixed|rolltemp|all   default: all
#   PYTHON_BIN=.venv/bin/python default: python
#   LOG_DIR=logs              default: logs
#   SLEEP_SECONDS=0           default: 0

MODE="${MODE:-all}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-logs}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0}"
DRY_RUN="${DRY_RUN:-0}"

S5_M="${S5_M:-21}"
N_TRAIN="${N_TRAIN:-15000000}"
N_VAL="${N_VAL:-5000}"
SUBSET_SIZE="${SUBSET_SIZE:-12000000}"
BANK_SEED="${BANK_SEED:-1337}"
TEACHER_SEED="${TEACHER_SEED:-20260417}"
PROMPT_BANK_DIR="${PROMPT_BANK_DIR:-data/s5_clean_prompt_bank_m21_n15000000_val5000}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-reruns/s5_m21_teacher20260417/out-s5-cot-m21-depth1-seed20260417}"
OUTPUT_ROOT="${OUTPUT_ROOT:-reruns/s5_random_suffix}"

SEEDS=(${SEEDS:-20260417 20260418 20260419})
ETAS=(${ETAS:-0.0 0.2})
BETAS=(${BETAS:-0.0 0.1 0.25 0.5 0.75 0.9 1.0})
TEMPS=(${TEMPS:-0.1 0.2 0.3 0.4})
LOSSES=(${LOSSES:-forward reverse})

if [[ "$MODE" != "all" && "$MODE" != "mixed" && "$MODE" != "rolltemp" ]]; then
  echo "MODE must be one of: all, mixed, rolltemp" >&2
  exit 2
fi

if [[ ! -d "$PROMPT_BANK_DIR" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "dry-run warning: missing prompt bank: $PROMPT_BANK_DIR" >&2
  else
    echo "missing prompt bank: $PROMPT_BANK_DIR" >&2
    exit 1
  fi
fi

if [[ ! -f "$TEACHER_CHECKPOINT/ckpt.pt" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "dry-run warning: missing teacher checkpoint ckpt.pt under: $TEACHER_CHECKPOINT" >&2
  else
    echo "missing teacher checkpoint ckpt.pt under: $TEACHER_CHECKPOINT" >&2
    exit 1
  fi
fi

mkdir -p "$LOG_DIR"

float_tag() {
  echo "${1/./p}"
}

launch_cmd() {
  local log_path="$1"
  shift
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'nohup env HYDRA_FULL_ERROR=1 %q' "$PYTHON_BIN"
    printf ' %q' "$@"
    printf ' > %q 2>&1 &\n' "$log_path"
  else
    nohup env HYDRA_FULL_ERROR=1 "$PYTHON_BIN" "$@" > "$log_path" 2>&1 &
    echo "launched pid=$! log=$log_path"
    if [[ "$SLEEP_SECONDS" != "0" ]]; then
      sleep "$SLEEP_SECONDS"
    fi
  fi
}

common_overrides() {
  local eta="$1"
  local seed="$2"
  printf '%s\n' \
    "experiment=s5_nail" \
    "task.s5_m=${S5_M}" \
    "task.n_train=${N_TRAIN}" \
    "task.n_val=${N_VAL}" \
    "task.bank_seed=${BANK_SEED}" \
    "task.teacher_seed=${TEACHER_SEED}" \
    "task.render_seed=${seed}" \
    "task.prompt_bank_dir=${PROMPT_BANK_DIR}" \
    "task.teacher_checkpoint=${TEACHER_CHECKPOINT}" \
    "task.subset_size=${SUBSET_SIZE}" \
    "task.teacher_law=random_suffix_after_error" \
    "task.eta=${eta}" \
    "task.random_suffix_noise.seed=${seed}" \
    "task.random_suffix_noise.apply_to=s5" \
    "task.random_suffix_noise.key_positions=semantic_key" \
    "task.random_suffix_noise.random_suffix_mode=valid_tokens" \
    "task.random_suffix_noise.keep_format_tokens=true" \
    "task.random_suffix_noise.coord_strategy=cyclic" \
    "task.teacher_signal=mc" \
    "optim.seed=${seed}" \
    "optim.eval_interval=5000"
}

if [[ "$MODE" == "all" || "$MODE" == "mixed" ]]; then
  for beta in "${BETAS[@]}"; do
    btag="$(float_tag "$beta")"
    for eta in "${ETAS[@]}"; do
      etag="$(float_tag "$eta")"
      for seed in "${SEEDS[@]}"; do
        run_name="s5-rsuffix-nail-mixed-beta${btag}-eta${etag}-seed${seed}"
        out_dir="${OUTPUT_ROOT}/nail_mixed_beta${btag}_eta${etag}_seed${seed}"
        log_path="${LOG_DIR}/s5_rsuffix_nail_mixed_beta${btag}_eta${etag}_seed${seed}.log"
        overrides=()
        while IFS= read -r override; do
          overrides+=("$override")
        done < <(common_overrides "$eta" "$seed")
        launch_cmd "$log_path" -m nanogpt.run \
          "${overrides[@]}" \
          "task.loss=mixed" \
          "task.kl_beta=${beta}" \
          "run.name=${run_name}" \
          "logging.wandb_run_name=${run_name}" \
          "run.out_dir=${out_dir}"
      done
    done
  done
fi

if [[ "$MODE" == "all" || "$MODE" == "rolltemp" ]]; then
  for loss in "${LOSSES[@]}"; do
    for temp in "${TEMPS[@]}"; do
      ttag="$(float_tag "$temp")"
      for eta in "${ETAS[@]}"; do
        etag="$(float_tag "$eta")"
        for seed in "${SEEDS[@]}"; do
          run_name="s5-rsuffix-nail-${loss}-rollt${ttag}-eta${etag}-seed${seed}"
          out_dir="${OUTPUT_ROOT}/rollout_temp_ablation/nail_${loss}/t${ttag}_eta${etag}_seed${seed}"
          log_path="${LOG_DIR}/s5_rsuffix_nail_${loss}_rollt${ttag}_eta${etag}_seed${seed}.log"
          overrides=()
          while IFS= read -r override; do
            overrides+=("$override")
          done < <(common_overrides "$eta" "$seed")
          launch_cmd "$log_path" -m nanogpt.run \
            "${overrides[@]}" \
            "task.loss=${loss}" \
            "task.rollout_temperature_override=${temp}" \
            "run.name=${run_name}" \
            "logging.wandb_run_name=${run_name}" \
            "run.out_dir=${out_dir}"
        done
      done
    done
  done
fi
