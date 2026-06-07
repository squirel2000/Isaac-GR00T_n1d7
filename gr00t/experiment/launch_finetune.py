# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Launch finetuning for N1.7 on "single node".
# This script tries to provide a similar user experience as current OSS.

import json
import os
from pathlib import Path

import numpy as np
import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run


# Make sure the user provided modality config is registered.
def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


def split_dataset_episodes(
    dataset_path: str,
    eval_split: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Split a dataset's episodes into reproducible (train_indices, val_indices).

    Reads ``meta/episodes.jsonl`` and shuffles with a fixed seed so the holdout is
    deterministic across runs (and identical on every rank).

    Args:
        dataset_path: Path to dataset root containing meta/episodes.jsonl.
        eval_split: Fraction of episodes held out for validation (0.0-1.0).
        seed: Random seed for the reproducible split.

    Returns:
        (train_indices, val_indices), each a sorted list of episode indices.
    """
    if not 0.0 < eval_split < 1.0:
        raise ValueError(
            f"eval_split must be in the open interval (0, 1) for an in-memory split, "
            f"got {eval_split}. Disable eval with --eval-strategy no, or provide a "
            f"dedicated --eval-dataset-path."
        )

    episodes_path = Path(dataset_path) / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Episodes file not found: {episodes_path}")

    num_episodes = 0
    with open(episodes_path, "r") as f:
        for line in f:
            if line.strip():
                num_episodes += 1
    if num_episodes < 2:
        raise ValueError(
            f"Need >= 2 episodes to hold out a validation split, found {num_episodes}. "
            f"Provide --eval-dataset-path or disable eval with --eval-strategy no."
        )

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(num_episodes)

    num_val = max(1, int(num_episodes * eval_split))
    val_indices = sorted(shuffled_indices[:num_val].tolist())
    train_indices = sorted(shuffled_indices[num_val:].tolist())

    print(
        f"[Dataset Split] Total episodes: {num_episodes}, "
        f"train: {len(train_indices)}, val: {len(val_indices)}"
    )
    return train_indices, val_indices


if __name__ == "__main__":
    # Set LOGURU_LEVEL environment variable if not already set (default: INFO)
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    # Use tyro for clean CLI
    ft_config = tyro.cli(FinetuneConfig, description=__doc__)
    from gr00t.data.embodiment_tags import EmbodimentTag

    ft_config.embodiment_tag = EmbodimentTag.resolve(ft_config.embodiment_tag)
    embodiment_tag = ft_config.embodiment_tag.value

    # all rank workers should register for the modality config
    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    dataset_paths = [path for path in ft_config.dataset_path.split(os.pathsep) if path]

    # Determine train/val episode indices for an optional in-memory holdout.
    train_episode_indices = None
    val_episode_indices = None
    if ft_config.eval_strategy == "no":
        print("[Eval] eval_strategy='no' -> evaluation disabled.")
    elif ft_config.eval_dataset_path is not None:
        print(f"[Eval] Using provided eval dataset: {ft_config.eval_dataset_path}")
    elif ft_config.eval_split > 0:
        if len(dataset_paths) > 1:
            raise ValueError(
                "In-memory eval split supports a single dataset path. Provide "
                "--eval-dataset-path when training on multiple dataset paths."
            )
        print(
            f"[Eval] No eval_dataset_path given -> in-memory split "
            f"(eval_split={ft_config.eval_split}, seed={ft_config.dataset_split_seed})"
        )
        train_episode_indices, val_episode_indices = split_dataset_episodes(
            dataset_paths[0],
            eval_split=ft_config.eval_split,
            seed=ft_config.dataset_split_seed,
        )
    else:
        raise ValueError(
            "eval_strategy is enabled but no eval source is available: pass "
            "--eval-dataset-path, or use a positive --eval-split, or set --eval-strategy no."
        )

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": dataset_paths,
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                        "val_dataset_path": ft_config.eval_dataset_path,
                        "train_episode_indices": train_episode_indices,
                        "val_episode_indices": val_episode_indices,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    # overwrite with finetune config supplied by the user
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    if ft_config.extra_augmentation_config:
        config.model.extra_augmentation_config = json.loads(ft_config.extra_augmentation_config)
    else:
        config.model.extra_augmentation_config = None

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.training.experiment_name = ft_config.experiment_name
    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = ft_config.wandb_project

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch

    config.training.eval_strategy = ft_config.eval_strategy
    config.training.eval_steps = ft_config.eval_steps
    config.training.eval_max_shards = ft_config.eval_max_shards
    config.training.save_best_eval_metric_name = ft_config.save_best_eval_metric_name
    config.training.save_best_eval_metric_greater_is_better = (
        ft_config.save_best_eval_metric_greater_is_better
    )
    config.training.early_stopping_patience = ft_config.early_stopping_patience
    config.training.early_stopping_min_delta = ft_config.early_stopping_min_delta

    config.training.save_only_model = ft_config.save_only_model
    config.training.skip_weight_loading = ft_config.skip_weight_loading

    run(config)
