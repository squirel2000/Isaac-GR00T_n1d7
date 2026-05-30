# SPDX-License-Identifier: Apache-2.0
"""
Modality config for the OpenArm-O6 (bimanual arm + LinkerHand dexhand) CanSorting dataset.

This file is passed to fine-tuning / eval via `--modality-config-path`. It registers a
ModalityConfig for EmbodimentTag.NEW_EMBODIMENT whose `modality_keys` MUST match the field
names in the dataset's `meta/modality.json`.

Dataset: /data/Gits/IsaacLab-GR00T/datasets/OpenArm_O6_CanSorting_dataset_0408
  state/action = 26 dims, split by meta/modality.json into:
      left_arm   [ 0: 7]   7 joints
      right_arm  [ 7:14]   7 joints
      left_hand  [14:20]   6 finger joints
      right_hand [20:26]   6 finger joints
  video: "camera" -> observation.images.camera (RGB 480x640)
  language: annotation.human.task_description ("pick and sort a green or orange can")

Action representation choice (tune freely):
  - arms  -> RELATIVE (delta from current joint state; N1.7 generalizes better with relative)
  - hands -> ABSOLUTE (open/closed targets, like a gripper)
  all NON_EEF (joint space, not end-effector) / DEFAULT format.
"""
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


openarm_o6_config = {
    # Video: current frame only; key must match the "video" entry in meta/modality.json
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["camera"],
    ),
    # State: current proprioception; keys must match "state" entries in meta/modality.json
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_arm", "right_arm", "left_hand", "right_hand"],
    ),
    # Action: 16-step prediction horizon; one ActionConfig per modality key (same order)
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["left_arm", "right_arm", "left_hand", "right_hand"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,  # left_arm joints
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,  # right_arm joints
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,  # left_hand fingers (open/closed targets)
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,  # right_hand fingers
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    # Language: task instruction
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(openarm_o6_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
