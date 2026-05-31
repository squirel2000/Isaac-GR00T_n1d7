# SPDX-License-Identifier: Apache-2.0
"""
OpenArm-O6 modality config: RIGHT arm + right hand only — single-arm tasks (13 dims).

Pass via `--modality-config-path examples/Openarm_LinkerHandO6/openarm_o6_config_right_only.py`.
Registers a subset of the dataset's meta/modality.json (right_arm[7:14] + right_hand[20:26]);
training on the full bimanual dataset is fine — only these limbs are read.

See openarm_o6_modalities.py for the shared builder and dataset layout.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from openarm_o6_modalities import build_openarm_o6_config


openarm_o6_config = build_openarm_o6_config(["right_arm", "right_hand"])

register_modality_config(openarm_o6_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
