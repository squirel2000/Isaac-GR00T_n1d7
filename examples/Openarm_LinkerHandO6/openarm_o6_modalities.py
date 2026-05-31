# SPDX-License-Identifier: Apache-2.0
"""
Shared modality-config builder for the OpenArm-O6 (bimanual arm + LinkerHand dexhand) embodiment.

The OpenArm-O6 dataset records a full bimanual setup in meta/modality.json:

    state/action (26 dims) split into:
        left_arm   [ 0: 7]   7 joints
        right_arm  [ 7:14]   7 joints
        left_hand  [14:20]   6 finger joints
        right_hand [20:26]   6 finger joints
    video:    "camera"  -> observation.images.camera (RGB 480x640)
    language: annotation.human.task_description ("pick and sort a green or orange can")

Not every task uses both arms. A single-arm task only activates one arm + the matching hand;
a bimanual task activates all four. Rather than duplicate a full ModalityConfig per case, this
module exposes `build_openarm_o6_config(active_limbs)`. The thin preset files in this directory
call it and register the result:

    openarm_o6_config.py             -> bimanual  (left_arm + right_arm + left_hand + right_hand)
    openarm_o6_config_right_only.py  -> right arm + right hand

For other combinations (e.g. left arm + hand, or arms only), call build_openarm_o6_config()
directly with your own limb list.

The file you pass to `--modality-config-path` selects the mode. The registered state/action
keys MUST be a subset of the dataset's meta/modality.json (a single-arm config trains fine on
the full bimanual dataset; only the listed limbs are read).

Action representation (tune freely):
    arms  -> RELATIVE (delta from current joint state; N1.7 generalizes better with relative)
    hands -> ABSOLUTE (open/closed finger targets, like a gripper)
  all NON_EEF (joint space, not end-effector) / DEFAULT format.

This module intentionally does NOT call register_modality_config() — importing it has no side
effects, so the preset files can share it without double-registering EmbodimentTag.NEW_EMBODIMENT.
"""

from collections.abc import Sequence

from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


# Canonical limb order — matches the [start:end] layout in meta/modality.json so the
# registered state/action keys line up with the dataset slices regardless of which subset
# is active.
CANONICAL_LIMB_ORDER = ["left_arm", "right_arm", "left_hand", "right_hand"]

# Per-limb action representation. Arms predict joint deltas (RELATIVE); hands predict
# absolute finger targets (ABSOLUTE), which behaves like an open/close gripper command.
_ARM_ACTION = ActionConfig(
    rep=ActionRepresentation.RELATIVE,
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
)
_HAND_ACTION = ActionConfig(
    rep=ActionRepresentation.ABSOLUTE,
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
)


def _action_config_for(limb: str) -> ActionConfig:
    if limb.endswith("_arm"):
        return _ARM_ACTION
    if limb.endswith("_hand"):
        return _HAND_ACTION
    raise ValueError(f"Cannot infer action representation for limb '{limb}'.")


def build_openarm_o6_config(
    active_limbs: Sequence[str],
    *,
    video_keys: Sequence[str] = ("camera",),
    action_horizon: int = 16,
) -> dict:
    """Build a modality config for the given subset of OpenArm-O6 limbs.

    Args:
        active_limbs: limbs to activate, e.g. ["right_arm", "right_hand"] for a single-arm
            task, or all four for bimanual. Any order/duplicates are accepted; the result is
            normalized to CANONICAL_LIMB_ORDER so state and action keys stay aligned with
            meta/modality.json.
        video_keys: camera keys (must match "video" entries in meta/modality.json).
        action_horizon: number of future steps to predict.

    Returns:
        A modality-config dict ready to pass to register_modality_config().
    """
    unknown = [limb for limb in active_limbs if limb not in CANONICAL_LIMB_ORDER]
    if unknown:
        raise ValueError(f"Unknown limb(s) {unknown}; valid limbs are {CANONICAL_LIMB_ORDER}.")

    # Normalize to canonical order (and de-duplicate) so the same keys appear, in the same
    # order, in both "state" and "action".
    active = set(active_limbs)
    limbs = [limb for limb in CANONICAL_LIMB_ORDER if limb in active]
    if not limbs:
        raise ValueError("active_limbs must contain at least one limb.")

    return {
        # Video: current frame only; key must match the "video" entry in meta/modality.json.
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=list(video_keys),
        ),
        # State: current proprioception for the active limbs.
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=limbs,
        ),
        # Action: `action_horizon`-step prediction horizon; one ActionConfig per limb (same order).
        "action": ModalityConfig(
            delta_indices=list(range(0, action_horizon)),
            modality_keys=limbs,
            action_configs=[_action_config_for(limb) for limb in limbs],
        ),
        # Language: task instruction.
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    }
