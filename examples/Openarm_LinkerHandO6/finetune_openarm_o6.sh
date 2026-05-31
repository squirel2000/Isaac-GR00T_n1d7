#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Fine-tune GR00T N1.7 on the OpenArm-O6 dataset.
#
# The OpenArm-O6 dataset records both arms + both hands. MODE selects which limbs the model sees:
#
#     MODE=bimanual    -> openarm_o6_config.py            (left_arm + right_arm + left_hand + right_hand)
#     MODE=right_only  -> openarm_o6_config_right_only.py (right_arm + right_hand)
#
# Every knob is env-overridable. Any extra CLI args are forwarded verbatim to the launcher
# (a leading `--` separator is optional and stripped). Examples:
#     # single right arm+hand, 50k steps on one GPU
#     MODE=right_only NUM_GPUS=1 MAX_STEPS=50000 \
#         bash examples/Openarm_LinkerHandO6/finetune_openarm_o6.sh
#
#     # both arms+hands, custom dataset/output
#     MODE=bimanual DATASET_PATH=/data/.../my_dataset OUTPUT_DIR=/data/.../run \
#         bash examples/Openarm_LinkerHandO6/finetune_openarm_o6.sh
#
#     # forward an extra launcher flag
#     MODE=right_only bash examples/Openarm_LinkerHandO6/finetune_openarm_o6.sh --tune-visual
#
# Gated backbone: GR00T-N1.7's Cosmos-Reason2-2B backbone is gated on Hugging Face. Export a
# token first so weight download succeeds:  export HF_TOKEN=hf_xxx
#
# Defaults target gr00t/experiment/launch_finetune.py for remote single-GPU (H100-class) runs.
# For local RTX 4090/5090 development, set:
#   LAUNCHER=gr00t/experiment/launch_finetune_asus.py
# OPTIM is only consumed by launch_finetune_asus.py. This wrapper defaults OPTIM to
# paged_adamw_8bit on that launcher path (install bitsandbytes first).
# USE_WANDB defaults to on (set USE_WANDB=0 to disable); NUM_GPUS defaults to 1.

set -x -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# --- Mode -> modality config preset ---------------------------------------------------------
MODE="${MODE:-bimanual}"
case "$MODE" in
    bimanual)   MODALITY_CONFIG_PATH="examples/Openarm_LinkerHandO6/openarm_o6_config.py" ;;
    right_only) MODALITY_CONFIG_PATH="examples/Openarm_LinkerHandO6/openarm_o6_config_right_only.py" ;;
    *)
        echo "Unknown MODE='$MODE'. Use one of: bimanual | right_only" >&2
        exit 1
        ;;
esac

# --- Paths (override via env) ---------------------------------------------------------------
DEFAULT_REMOTE_LAUNCHER="gr00t/experiment/launch_finetune.py"
LAUNCHER="${LAUNCHER:-$DEFAULT_REMOTE_LAUNCHER}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-nvidia/GR00T-N1.7-3B}"
DATASET_PATH="${DATASET_PATH:-/data/Gits/IsaacLab-GR00T/datasets/OpenArm_O6_CanSorting_dataset_0408}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-NEW_EMBODIMENT}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/Gits/IsaacLab-GR00T/artifacts/gr00t_openarm_o6_${MODE}_$(date +%m%d)}"
mkdir -p "$OUTPUT_DIR"

LAUNCHER_NAME="$(basename "$LAUNCHER")"
IS_ASUS_LAUNCHER=0
LAUNCHER_PROFILE="custom"
case "$LAUNCHER_NAME" in
    launch_finetune.py)
        LAUNCHER_PROFILE="remote_default"
        ;;
    launch_finetune_asus.py)
        IS_ASUS_LAUNCHER=1
        LAUNCHER_PROFILE="local_asus"
        ;;
    *)
        echo "[OpenArm-O6] WARNING: Unrecognized launcher '$LAUNCHER'."
        echo "[OpenArm-O6] WARNING: CUDA policy falls back to default single-GPU device 0."
        ;;
esac

# --- Distributed / runtime ------------------------------------------------------------------
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29505}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- Training hyperparameters (conservative single-GPU defaults) -----------------------------
MAX_STEPS="${MAX_STEPS:-50000}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
STATE_DROPOUT_PROB="${STATE_DROPOUT_PROB:-0.2}"
SHARD_SIZE="${SHARD_SIZE:-2048}"
EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-0.1}"
NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-100000}"
WANDB_PROJECT="${WANDB_PROJECT:-finetune-gr00t-n1d7}"

# OPTIM only affects launch_finetune_asus.py, so set a launcher-specific default here.
if [ "$IS_ASUS_LAUNCHER" = "1" ]; then
    export OPTIM="${OPTIM:-paged_adamw_8bit}"
    OPTIM_STATUS="$OPTIM (used by $LAUNCHER_NAME)"
else
    OPTIM_STATUS="${OPTIM:-N/A} (ignored by $LAUNCHER_NAME)"
    if [ -n "${OPTIM:-}" ]; then
        echo "[OpenArm-O6] NOTE: OPTIM='$OPTIM' is ignored when using $LAUNCHER_NAME."
    fi
fi

# Build the argument list once (keeps the launch command readable).
# NOTE: flags use the hyphen form to match the launcher's USAGE examples; tyro 0.9.17 (pinned
# in pyproject.toml) accepts either the hyphen or underscore form for dataclass fields.
TRAIN_ARGS=(
    "$LAUNCHER"
    --base-model-path "$BASE_MODEL_PATH"
    --dataset-path "$DATASET_PATH"
    --modality-config-path "$MODALITY_CONFIG_PATH"
    --embodiment-tag "$EMBODIMENT_TAG"
    --num-gpus "$NUM_GPUS"
    --output-dir "$OUTPUT_DIR"
    --max-steps "$MAX_STEPS"
    --save-steps "$SAVE_STEPS"
    --save-total-limit "$SAVE_TOTAL_LIMIT"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
    --learning-rate "$LEARNING_RATE"
    --warmup-ratio "$WARMUP_RATIO"
    --weight-decay "$WEIGHT_DECAY"
    --state-dropout-prob "$STATE_DROPOUT_PROB"
    --shard-size "$SHARD_SIZE"
    --episode-sampling-rate "$EPISODE_SAMPLING_RATE"
    --num-shards-per-epoch "$NUM_SHARDS_PER_EPOCH"
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
    --wandb-project "$WANDB_PROJECT"
)

# Optional args, only added when explicitly set via env.
# USE_WANDB defaults to enabled; set USE_WANDB=0 to disable logging.
if [ "${USE_WANDB:-1}" = "1" ]; then
    TRAIN_ARGS+=(--use-wandb)
fi
if [ -n "${EXPERIMENT_NAME:-}" ]; then
    TRAIN_ARGS+=(--experiment-name "$EXPERIMENT_NAME")
fi
if [ "${SAVE_ONLY_MODEL:-0}" = "1" ]; then
    TRAIN_ARGS+=(--save-only-model)
fi
# Forward any remaining CLI args to the launcher (drop an optional leading `--` separator).
if [ "${1:-}" = "--" ]; then
    shift
fi
TRAIN_ARGS+=("$@")

CUDA_POLICY="multi_gpu_torchrun"
CUDA_VISIBLE_DEVICES_EFFECTIVE="${CUDA_VISIBLE_DEVICES:-<unset>}"
if [ "$NUM_GPUS" = "1" ]; then
    if [ "$IS_ASUS_LAUNCHER" = "1" ]; then
        # Local Asus launcher: force single-device execution on GPU 0.
        export CUDA_VISIBLE_DEVICES=0
        CUDA_POLICY="asus_single_gpu_fixed_to_0"
    elif [ "$LAUNCHER_NAME" = "launch_finetune.py" ]; then
        # Remote default launcher: allow either GPU 0 or GPU 1 for single-GPU runs.
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
        if [ "$CUDA_VISIBLE_DEVICES" != "0" ] && [ "$CUDA_VISIBLE_DEVICES" != "1" ]; then
            echo "For LAUNCHER=$LAUNCHER with NUM_GPUS=1, CUDA_VISIBLE_DEVICES must be 0 or 1." >&2
            exit 1
        fi
        CUDA_POLICY="remote_single_gpu_allow_0_or_1"
    else
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
        CUDA_POLICY="custom_single_gpu_default_to_0"
    fi
    CUDA_VISIBLE_DEVICES_EFFECTIVE="$CUDA_VISIBLE_DEVICES"
fi

# Record the resolved parameters for reproducibility.
PARAMS_FILE="$OUTPUT_DIR/finetune_params.txt"
{
    echo "Execution time: $(date +'%Y%m%d_%H%M')"
    echo "MODE: $MODE"
    echo "LAUNCHER: $LAUNCHER"
    echo "LAUNCHER_PROFILE: $LAUNCHER_PROFILE"
    echo "OPTIM: $OPTIM_STATUS"
    echo "GRAD_CKPT: ${GRAD_CKPT:-0}"
    echo "USE_WANDB: ${USE_WANDB:-1}"
    echo "NUM_GPUS: $NUM_GPUS"
    echo "CUDA_POLICY: $CUDA_POLICY"
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES_EFFECTIVE"
    echo "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
    echo "-------------------------"
    echo "${TRAIN_ARGS[@]:1}" | sed 's/ --/\n--/g'
} >"$PARAMS_FILE"

echo "[OpenArm-O6] MODE=$MODE launcher=$LAUNCHER profile=$LAUNCHER_PROFILE output=$OUTPUT_DIR"
echo "[OpenArm-O6] NUM_GPUS=$NUM_GPUS CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_EFFECTIVE"
echo "[OpenArm-O6] OPTIM=$OPTIM_STATUS"

if [ "$NUM_GPUS" = "1" ]; then
    # NUM_GPUS=1 path: launch a single python process (no torchrun).
    exec uv run python "${TRAIN_ARGS[@]}"
fi

exec uv run torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" "${TRAIN_ARGS[@]}"
