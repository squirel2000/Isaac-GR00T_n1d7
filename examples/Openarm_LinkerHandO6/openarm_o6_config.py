# SPDX-License-Identifier: Apache-2.0
"""
OpenArm-O6 modality config: BIMANUAL — both arms + both hands (26 dims).

Pass via `--modality-config-path examples/Openarm_LinkerHandO6/openarm_o6_config.py`.
Use this for tasks that move both arms. For single-arm tasks, use the
`openarm_o6_config_right_only.py` preset instead.

See openarm_o6_modalities.py for the shared builder, the dataset layout, and the
action-representation rationale (arms RELATIVE, hands ABSOLUTE).
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from openarm_o6_modalities import build_openarm_o6_config


openarm_o6_config = build_openarm_o6_config(["left_arm", "right_arm", "left_hand", "right_hand"])

register_modality_config(openarm_o6_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
