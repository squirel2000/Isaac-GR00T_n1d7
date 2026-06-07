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

"""Tests for the held-out action-MSE eval + early-stop mechanism.

- CPU: EvalMetricEarlyStoppingCallback patience / min_delta / direction logic.
- GPU: end-to-end that eval emits ``eval_action_mse`` and a best checkpoint is
  saved. Mirrors ``test_experiment_run_single_gpu`` but with eval enabled. Runs on
  a single GPU (e.g. an RTX 4090); weights are skipped, only metadata is fetched.
"""

import json

import pytest
from test_support.runtime import get_root, resolve_libero_demo_dataset_path
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from gr00t.experiment.utils import EvalMetricEarlyStoppingCallback


REPO_ROOT = get_root()
EMBODIMENT_TAG = "libero_sim"
MODEL_REPO_ID = "nvidia/GR00T-N1.7-3B"
METRIC = "eval_action_mse"


def _feed(callback, values):
    """Feed a sequence of metric values; return the step at which training stops."""
    args = TrainingArguments(output_dir="/tmp/_es_test")
    for i, v in enumerate(values):
        state = TrainerState()
        state.global_step = i
        control = TrainerControl()
        callback.on_evaluate(args, state, control, metrics={METRIC: v}, model=None)
        if control.should_training_stop:
            return i
    return None


def test_early_stop_triggers_after_patience():
    """Lower-is-better: stop once the metric fails to improve `patience` times."""
    cb = EvalMetricEarlyStoppingCallback(METRIC, patience=2, greater_is_better=False)
    # 1.0 (best) -> 0.5 (best) -> 0.6 (stale 1) -> 0.7 (stale 2 -> stop)
    assert _feed(cb, [1.0, 0.5, 0.6, 0.7]) == 3


def test_early_stop_resets_on_improvement():
    """An improvement resets the stale counter, so a later dip does not stop early."""
    cb = EvalMetricEarlyStoppingCallback(METRIC, patience=2, greater_is_better=False)
    # 1.0 -> 1.1 (stale1) -> 0.4 (improve, reset) -> 0.5 (stale1) : never reaches 2
    assert _feed(cb, [1.0, 1.1, 0.4, 0.5]) is None


def test_early_stop_min_delta_requires_meaningful_gain():
    """A change smaller than min_delta does not count as an improvement."""
    cb = EvalMetricEarlyStoppingCallback(
        METRIC, patience=1, min_delta=0.1, greater_is_better=False
    )
    # 1.0 (best) -> 0.95 (gain 0.05 < 0.1 -> stale1 == patience -> stop)
    assert _feed(cb, [1.0, 0.95]) == 1


def test_early_stop_disabled_when_patience_zero():
    cb = EvalMetricEarlyStoppingCallback(METRIC, patience=0, greater_is_better=False)
    assert _feed(cb, [1.0, 2.0, 3.0, 4.0]) is None


def test_early_stop_greater_is_better():
    """Higher-is-better direction stops when the metric stops rising."""
    cb = EvalMetricEarlyStoppingCallback(METRIC, patience=1, greater_is_better=True)
    # 0.1 (best) -> 0.2 (best) -> 0.15 (stale1 == patience -> stop)
    assert _feed(cb, [0.1, 0.2, 0.15]) == 2


@pytest.mark.gpu
@pytest.mark.timeout(600, func_only=True)
def test_experiment_run_single_gpu_eval(tmp_path, monkeypatch):
    """Run experiment.run() with held-out eval enabled and assert the metric path works.

    Verifies the two things the commit's mechanism depends on:
      1. ``eval_action_mse`` is actually computed and logged (the bug was that
         compute_metrics never ran, so the metric never appeared).
      2. ``BestMetricCheckpointCallback`` saves a ``*-best-eval_action_mse_*`` dir.
    """
    import numpy as np
    import torch

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    from gr00t.configs.base_config import get_default_config
    from gr00t.experiment.experiment import run
    from gr00t.experiment.launch_finetune import split_dataset_episodes

    dataset_path = resolve_libero_demo_dataset_path(REPO_ROOT)
    output_dir = tmp_path / "experiment_output_eval"

    # In-memory holdout split, exactly as launch_finetune.py computes it.
    train_idx, val_idx = split_dataset_episodes(str(dataset_path), eval_split=0.4, seed=42)
    assert len(train_idx) > 0 and len(val_idx) > 0

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [str(dataset_path)],
                        "mix_ratio": 1.0,
                        "embodiment_tag": EMBODIMENT_TAG,
                        "train_episode_indices": train_idx,
                        "val_episode_indices": val_idx,
                    }
                ],
                "video_backend": "torchcodec",
                "shard_size": 64,
                "num_shards_per_epoch": 1,
                "multiprocessing_context": "fork",
            },
        }
    )

    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True
    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.tune_llm = False
    config.model.tune_visual = False
    config.model.tune_projector = True
    config.model.tune_diffusion_model = True

    config.training.start_from_checkpoint = MODEL_REPO_ID
    config.training.skip_weight_loading = True
    config.training.output_dir = str(output_dir)
    config.training.max_steps = 2
    config.training.save_steps = 2
    config.training.global_batch_size = 2
    config.training.eval_batch_size = 2
    config.training.num_gpus = 1
    config.training.dataloader_num_workers = 0
    config.training.use_wandb = False
    config.training.optim = "adamw_torch"
    config.training.bf16 = True
    config.training.tf32 = True
    config.training.fp16 = False
    config.training.gradient_checkpointing = False
    config.training.use_ddp = False

    # Enable held-out eval + best-checkpoint saving (the mechanism under test).
    config.training.eval_strategy = "steps"
    config.training.eval_steps = 1
    config.training.save_best_eval_metric_name = METRIC
    config.training.save_best_eval_metric_greater_is_better = False

    run(config)

    # 1. The eval metric was computed and logged into trainer_state.
    checkpoint_dirs = list(output_dir.glob("checkpoint-*"))
    assert checkpoint_dirs, f"No checkpoint saved: {list(output_dir.iterdir())}"
    trainer_state_path = next(
        (c / "trainer_state.json" for c in checkpoint_dirs if (c / "trainer_state.json").exists()),
        None,
    )
    assert trainer_state_path is not None, "trainer_state.json missing from all checkpoints"
    log_history = json.loads(trainer_state_path.read_text()).get("log_history", [])
    eval_entries = [e for e in log_history if METRIC in e]
    assert eval_entries, f"'{METRIC}' never logged; eval metric path is broken: {log_history}"
    assert np.isfinite(eval_entries[-1][METRIC]), f"{METRIC} not finite: {eval_entries[-1]}"

    # 2. A best-metric checkpoint was saved by BestMetricCheckpointCallback.
    best_dirs = list(output_dir.glob(f"checkpoint-*-best-{METRIC}_*"))
    assert best_dirs, f"No best-{METRIC} checkpoint saved: {[p.name for p in output_dir.iterdir()]}"

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
