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

import numpy as np
import torch
from tqdm import tqdm

from gr00t.configs.base_config import Config
from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.stats import generate_rel_stats, generate_stats
from gr00t.experiment.dist_utils import barrier


class DatasetFactory:
    """
    Factory class for building training datasets. Model-agnostic.
    """

    def __init__(self, config: Config):
        self.config = config

    def build(
        self, processor: BaseProcessor
    ) -> tuple[ShardedMixtureDataset, ShardedMixtureDataset | None]:
        """Build the dataset. Returns a tuple of (train_dataset, eval_dataset).

        When ``training.eval_strategy != "no"`` an eval mixture is also built from
        either a dedicated ``val_dataset_path`` or an in-memory holdout selected by
        ``val_episode_indices`` (the complement of ``train_episode_indices``).
        """
        all_train_datasets = []
        all_train_weights = []
        all_eval_datasets = []
        all_eval_weights = []

        for dataset_spec in tqdm(
            self.config.data.datasets,
            total=len(self.config.data.datasets),
            desc="Initializing datasets",
        ):
            embodiment_tag = dataset_spec.embodiment_tag
            assert embodiment_tag is not None, "Embodiment tag is required"
            assert self.config.data.mode == "single_turn", "Only single turn mode is supported"

            # In-memory train/val split indices are computed from a single dataset's
            # episode count, so they are only valid for a single-path spec. Applying
            # them across multiple paths would silently mismatch episode counts.
            if (
                dataset_spec.train_episode_indices is not None
                or dataset_spec.val_episode_indices is not None
            ) and len(dataset_spec.dataset_paths) > 1:
                raise ValueError(
                    "In-memory train/val episode split supports a single dataset path "
                    f"per spec, but got {len(dataset_spec.dataset_paths)} paths. Provide "
                    "a dedicated val_dataset_path for multi-path specs."
                )

            datasets = []
            for dataset_path in dataset_spec.dataset_paths:
                if torch.distributed.is_initialized():
                    if torch.distributed.get_rank() == 0:
                        generate_stats(dataset_path)
                        generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
                else:
                    generate_stats(dataset_path)
                    generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
                barrier()
                dataset = ShardedSingleStepDataset(
                    dataset_path=dataset_path,
                    embodiment_tag=EmbodimentTag(embodiment_tag),
                    modality_configs=self.config.data.modality_configs[embodiment_tag],
                    video_backend=self.config.data.video_backend,
                    shard_size=self.config.data.shard_size,
                    episode_sampling_rate=self.config.data.episode_sampling_rate,
                    seed=self.config.data.seed,
                    allow_padding=self.config.data.allow_padding,
                    episode_indices=dataset_spec.train_episode_indices,
                )
                datasets.append(dataset)
            dataset_lengths = np.array([len(dataset) for dataset in datasets])
            dataset_relative_lengths = dataset_lengths / dataset_lengths.sum()
            for dataset, relative_length in zip(datasets, dataset_relative_lengths):
                weight = relative_length * dataset_spec.mix_ratio
                all_train_datasets.append(dataset)
                all_train_weights.append(weight)

            # Build eval datasets (only when evaluation is enabled). Two sources:
            #   1. A dedicated val_dataset_path.
            #   2. In-memory train/val split: reuse dataset_paths but keep only
            #      val_episode_indices (full sampling rate so the holdout is exact).
            if self.config.training.eval_strategy != "no":
                val_path = dataset_spec.val_dataset_path
                if val_path:
                    eval_paths = [val_path]
                elif dataset_spec.val_episode_indices is not None:
                    eval_paths = dataset_spec.dataset_paths
                else:
                    eval_paths = []

                for eval_path in eval_paths:
                    if torch.distributed.is_initialized():
                        if torch.distributed.get_rank() == 0:
                            generate_stats(eval_path)
                            generate_rel_stats(eval_path, EmbodimentTag(embodiment_tag))
                    else:
                        generate_stats(eval_path)
                        generate_rel_stats(eval_path, EmbodimentTag(embodiment_tag))
                    barrier()
                    eval_dataset = ShardedSingleStepDataset(
                        dataset_path=eval_path,
                        embodiment_tag=EmbodimentTag(embodiment_tag),
                        modality_configs=self.config.data.modality_configs[embodiment_tag],
                        video_backend=self.config.data.video_backend,
                        shard_size=self.config.data.shard_size,
                        episode_sampling_rate=1.0,
                        seed=self.config.data.seed,
                        allow_padding=self.config.data.allow_padding,
                        episode_indices=dataset_spec.val_episode_indices,
                    )
                    all_eval_datasets.append(eval_dataset)
                    all_eval_weights.append(dataset_spec.mix_ratio)

        train_mixture = ShardedMixtureDataset(
            datasets=all_train_datasets,
            weights=all_train_weights,
            processor=processor,
            seed=self.config.data.seed,
            training=True,
            num_shards_per_epoch=self.config.data.num_shards_per_epoch,
            override_pretraining_statistics=self.config.data.override_pretraining_statistics,
        )

        eval_mixture = None
        if all_eval_datasets:
            eval_mixture = ShardedMixtureDataset(
                datasets=all_eval_datasets,
                weights=all_eval_weights,
                processor=processor,
                seed=self.config.data.seed,
                training=False,
                override_pretraining_statistics=False,
                eval_max_shards=self.config.training.eval_max_shards,
            )

        return train_mixture, eval_mixture
