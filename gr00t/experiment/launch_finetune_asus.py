#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""
Low-VRAM GR00T N1.7 fine-tuning launcher tuned for a single RTX 5090 (32 GB, Blackwell sm_120).

WHY THIS EXISTS
---------------
The shipped `gr00t/experiment/launch_finetune.py` hard-codes the optimizer to
`adamw_torch`. Full diffusion-head fine-tuning trains ~1.62B params (51%), and AdamW
needs ~13 GB of fp32 momentum state (exp_avg + exp_avg_sq) plus a transient copy during
`optimizer.step()`. That alone overflows 32 GB on the very first optimizer step,
regardless of batch size — which is why the README recommends 40 GB+ cards.

This launcher is a thin, non-invasive wrapper around the exact same config pipeline, with
three changes that make it fit on a 32 GB card:
  1. optim -> "adafactor" by default (factored 2nd moment => optimizer state drops 13 GB -> ~10 MB)
  2. recommends 0 dataloader workers (the USAGE example below and finetune_openarm_o6.sh both
     pass --dataloader-num-workers 0) to avoid a fork + AV1-video-decode (torchcodec/ffmpeg)
     DEADLOCK that otherwise freezes training mid-run (GPU util drops to 1% while VRAM stays
     fully allocated; main proc stuck in futex_wait, workers in do_poll).
  3. exposes optim / gradient-checkpointing via env so you can experiment.

MEASURED on this box: per-device batch 16 (x grad-accum 2 = effective 32) peaks at ~25 GB,
~1.3-1.6 it/s. Loss on the 5-episode demo set drops 1.17 -> ~0.04; open-loop action MSE
improves ~59x vs a 60-step checkpoint.

USAGE
-----
    # Run from the repo root with the uv env synced (see docs/README_RTX5090.md). This example
    # fine-tunes the SINGLE RIGHT arm+hand (right_only); swap the modality config for bimanual.
    export CUDA_VISIBLE_DEVICES=0
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation
    export OPTIM=paged_adamw_8bit                             # verified on this box (needs bitsandbytes)
    nohup uv run python gr00t/experiment/launch_finetune_asus.py \
        --base-model-path nvidia/GR00T-N1.7-3B \
        --dataset-path /data/Gits/IsaacLab-GR00T/datasets/OpenArm_O6_CanSorting_dataset_0408 \
        --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path examples/Openarm_LinkerHandO6/openarm_o6_config_right_only.py \
        --num-gpus 1 \
        --output-dir /data/Gits/IsaacLab-GR00T/artifacts/gr00t_openarm_O6_CanSorting_0408_right_only_50k \
        --max-steps 50000 \
        --save-steps 2000 \
        --save-total-limit 2 \
        --global-batch-size 16 \
        --gradient-accumulation-steps 2 \
        --dataloader-num-workers 0 \
        --use-wandb --wandb-project gr00t-openarm-o6_right_only_50k \
        > /data/Gits/IsaacLab-GR00T/artifacts/gr00t_openarm_O6_CanSorting_0408_right_only_50k.log 2>&1 &

    # Bimanual instead: --modality-config-path examples/Openarm_LinkerHandO6/openarm_o6_config.py

Env knobs:
    OPTIM         optimizer name (default: adafactor). "paged_adamw_8bit" is the verified pick on
                  this box (needs `uv pip install bitsandbytes`); closer-to-default AdamW dynamics.
    GRAD_CKPT     set to "1" to enable gradient checkpointing (lower activation memory,
                  ~20-30% slower; not needed at batch 16 on a 5090).

Resuming: HF Trainer auto-resumes from the highest `checkpoint-N` already in --output-dir,
so re-running the same command after an interruption (e.g. a reboot) continues where it left
off. Use a FRESH --output-dir for a clean run from step 0.
"""
import json
import os
import sys
from pathlib import Path

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run


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


def main():
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"

    ft = tyro.cli(FinetuneConfig, description=__doc__)
    if (
        "--dataloader-num-workers" not in sys.argv
        and "--dataloader_num_workers" not in sys.argv
    ):
        ft.dataloader_num_workers = 0

    from gr00t.data.embodiment_tags import EmbodimentTag

    ft.embodiment_tag = EmbodimentTag.resolve(ft.embodiment_tag)
    embodiment_tag = ft.embodiment_tag.value

    if ft.modality_config_path is not None:
        load_modality_config(ft.modality_config_path)

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [ft.dataset_path],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    # --- model tuning flags (defaults: freeze LLM+visual, train projector+diffusion head) ---
    config.model.tune_llm = ft.tune_llm
    config.model.tune_visual = ft.tune_visual
    config.model.tune_projector = ft.tune_projector
    config.model.tune_diffusion_model = ft.tune_diffusion_model
    config.model.state_dropout_prob = ft.state_dropout_prob
    config.model.random_rotation_angle = ft.random_rotation_angle
    config.model.color_jitter_params = ft.color_jitter_params
    config.model.extra_augmentation_config = (
        json.loads(ft.extra_augmentation_config) if ft.extra_augmentation_config else None
    )
    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    # --- training config ---
    config.training.experiment_name = ft.experiment_name
    config.training.start_from_checkpoint = ft.base_model_path

    # >>> LOW-VRAM OVERRIDES (the whole point of this wrapper) <<<
    config.training.optim = os.environ.get("OPTIM", "adafactor")
    if os.environ.get("GRAD_CKPT", "0") == "1":
        config.training.gradient_checkpointing = True
    # <<< END OVERRIDES >>>

    config.training.global_batch_size = ft.global_batch_size
    # 0 workers (set by the shell wrapper / USAGE example) = no fork = no AV1-decode deadlock;
    # raise via --dataloader-num-workers only with a fork-safe multiprocessing_context.
    config.training.dataloader_num_workers = ft.dataloader_num_workers
    config.training.learning_rate = ft.learning_rate
    config.training.gradient_accumulation_steps = ft.gradient_accumulation_steps
    config.training.output_dir = ft.output_dir
    config.training.save_steps = ft.save_steps
    config.training.save_total_limit = ft.save_total_limit
    config.training.num_gpus = ft.num_gpus
    config.training.use_wandb = ft.use_wandb
    config.training.max_steps = ft.max_steps
    config.training.weight_decay = ft.weight_decay
    config.training.warmup_ratio = ft.warmup_ratio
    config.training.wandb_project = ft.wandb_project

    config.data.shard_size = ft.shard_size
    config.data.episode_sampling_rate = ft.episode_sampling_rate
    config.data.num_shards_per_epoch = ft.num_shards_per_epoch

    config.training.save_only_model = ft.save_only_model
    config.training.skip_weight_loading = ft.skip_weight_loading

    print(
        f"[rtx5090 low-vram] optim={config.training.optim} "
        f"gradient_checkpointing={config.training.gradient_checkpointing} "
        f"per_device_batch={ft.global_batch_size // ft.num_gpus} "
        f"grad_accum={ft.gradient_accumulation_steps} "
        f"effective_batch={(ft.global_batch_size // ft.num_gpus) * ft.gradient_accumulation_steps} "
        f"num_workers={ft.dataloader_num_workers}"
    )
    run(config)


if __name__ == "__main__":
    main()
