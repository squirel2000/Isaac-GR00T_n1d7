#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Open-loop evaluation for an OpenArm-O6 GR00T N1.7 checkpoint.
#
# The active limbs are baked into the checkpoint at train time, so eval does not need a
# modality config. MODE only labels the default plot path.
#
# Example:
#     MODE=right_only \
#     CHECKPOINT_PATH=/data/.../gr00t_openarm_o6_right_only_0531/checkpoint-50000 \
#         bash examples/Openarm_LinkerHandO6/openloop_eval_openarm_o6.sh

set -x -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

MODE="${MODE:-bimanual}"
DATASET_PATH="${DATASET_PATH:-/data/Gits/IsaacLab-GR00T/datasets/OpenArm_O6_CanSorting_dataset_0408}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to a trained checkpoint dir}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-NEW_EMBODIMENT}"
TRAJ_IDS="${TRAJ_IDS:-0}"
ACTION_HORIZON="${ACTION_HORIZON:-16}"
STEPS="${STEPS:-1000}"
DENOISING_STEPS="${DENOISING_STEPS:-16}"
SAVE_PLOT_PATH="${SAVE_PLOT_PATH:-}"
read -r -a TRAJ_ID_ARGS <<<"$TRAJ_IDS"

EVAL_ARGS=(
    --dataset-path "$DATASET_PATH"
    --model-path "$CHECKPOINT_PATH"
    --embodiment-tag "$EMBODIMENT_TAG"
    --action-horizon "$ACTION_HORIZON"
    --steps "$STEPS"
    --denoising-steps "$DENOISING_STEPS"
)

# A space-separated TRAJ_IDS value ("0 1 2") becomes multiple --traj-ids values.
EVAL_ARGS+=(--traj-ids "${TRAJ_ID_ARGS[@]}")

if [ -n "${SAVE_PLOT_PATH:-}" ]; then
    mkdir -p "$(dirname "$SAVE_PLOT_PATH")"
    EVAL_ARGS+=(--save-plot-path "$SAVE_PLOT_PATH")
elif [ "${#TRAJ_ID_ARGS[@]}" = "1" ]; then
    SAVE_PLOT_PATH="$CHECKPOINT_PATH/open_loop_eval/openarm_o6_${MODE}_traj_${TRAJ_ID_ARGS[0]}.jpeg"
    mkdir -p "$(dirname "$SAVE_PLOT_PATH")"
    EVAL_ARGS+=(--save-plot-path "$SAVE_PLOT_PATH")
fi

uv run python gr00t/eval/open_loop_eval.py "${EVAL_ARGS[@]}"
