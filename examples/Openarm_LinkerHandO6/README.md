# Fine-tuning GR00T N1.7 on OpenArm-O6

OpenArm-O6 is a **bimanual** setup: two 7-DoF arms plus two 6-DoF LinkerHand dexhands.
The dataset records all four limbs, but tasks differ — some move a **single** arm+hand, some
move **both**. This example supports both cases through two interchangeable modality presets.

## Dataset layout

`meta/modality.json` splits the 26-dim state/action vector into four limbs:

| key          | slice      | DoF | action representation |
|--------------|------------|-----|-----------------------|
| `left_arm`   | `[0:7]`    | 7   | RELATIVE (joint delta) |
| `right_arm`  | `[7:14]`   | 7   | RELATIVE (joint delta) |
| `left_hand`  | `[14:20]`  | 6   | ABSOLUTE (finger target) |
| `right_hand` | `[20:26]`  | 6   | ABSOLUTE (finger target) |

- video: `camera` → `observation.images.camera` (RGB 480×640)
- language: `annotation.human.task_description`

Arms use **RELATIVE** actions (N1.7 generalizes better with joint deltas); hands use
**ABSOLUTE** finger targets (behaves like an open/close gripper). All are joint-space
(`NON_EEF`) / `DEFAULT` format.

## Modality presets (which limbs are active)

| MODE         | preset file                       | active limbs                                    | dims |
|--------------|-----------------------------------|-------------------------------------------------|------|
| `bimanual`   | `openarm_o6_config.py`            | `left_arm` + `right_arm` + `left_hand` + `right_hand` | 26 |
| `right_only` | `openarm_o6_config_right_only.py` | `right_arm` + `right_hand`                      | 13   |

Both are thin wrappers around [`build_openarm_o6_config()`](openarm_o6_modalities.py), which
assembles the `ModalityConfig` from a list of limbs (normalized to the canonical order so
state/action keys stay aligned with `modality.json`). A single-arm preset registers a **subset**
of the dataset's keys — training on the full bimanual dataset is fine; only the listed limbs are
read. For other combinations (left arm + hand, arms only, …), call the builder with your own
limb list, e.g. `build_openarm_o6_config(["left_arm", "left_hand"])`.

> The file you pass to `--modality-config-path` selects the mode. Exactly one preset registers
> `EmbodimentTag.NEW_EMBODIMENT` per run; the shared builder module never registers on import,
> so the presets can share it safely.

## Fine-tuning

`finetune_openarm_o6.sh` maps `MODE` to the matching preset and fills in OpenArm-O6
defaults. It uses `gr00t/experiment/launch_finetune.py` by default for remote single-GPU
fine-tuning (for example, H100). Everything is env-overridable; anything after `--` is
forwarded to the launcher.

```bash
# Gated backbone (Cosmos-Reason2-2B): export a HF token first.
export HF_TOKEN=hf_xxx

# Remote default launcher on GPU 0.
MODE=right_only NUM_GPUS=1 MAX_STEPS=50000 \
    bash examples/Openarm_LinkerHandO6/finetune_openarm_o6.sh

# Remote default launcher on GPU 1 (optional, single-GPU mode).
CUDA_VISIBLE_DEVICES=1 MODE=right_only NUM_GPUS=1 MAX_STEPS=50000 \
    bash examples/Openarm_LinkerHandO6/finetune_openarm_o6.sh

# Both arms + hands, custom dataset/output.
MODE=bimanual \
    DATASET_PATH=/data/.../my_dataset \
    OUTPUT_DIR=/data/.../run \
    bash examples/Openarm_LinkerHandO6/finetune_openarm_o6.sh

# Local RTX 4090/5090 development profile.
LAUNCHER=gr00t/experiment/launch_finetune_asus.py MODE=right_only \
    bash examples/Openarm_LinkerHandO6/finetune_openarm_o6.sh
```

Common env knobs (with defaults): `MODE=bimanual`, `NUM_GPUS=1`, `MAX_STEPS=50000`,
`SAVE_STEPS=2000`, `GLOBAL_BATCH_SIZE=16`, `GRADIENT_ACCUMULATION_STEPS=2`,
`DATALOADER_NUM_WORKERS=0`, `BASE_MODEL_PATH=nvidia/GR00T-N1.7-3B`,
`LAUNCHER=gr00t/experiment/launch_finetune.py`, `USE_WANDB=1`,
`OUTPUT_DIR=/data/.../gr00t_openarm_o6_<MODE>_<MMDD>`.

### Launcher-specific behavior

- `LAUNCHER=gr00t/experiment/launch_finetune.py` (default):
  - designed for remote fine-tuning
  - with `NUM_GPUS=1`, `CUDA_VISIBLE_DEVICES` is validated as `0` or `1` (default `0`)
  - `OPTIM` is ignored on this launcher path
- `LAUNCHER=gr00t/experiment/launch_finetune_asus.py`:
  - intended for local RTX 4090/5090 development
  - with `NUM_GPUS=1`, `CUDA_VISIBLE_DEVICES` is forced to `0`
  - `OPTIM` is consumed here and defaults to `paged_adamw_8bit` in the wrapper

For the Asus profile, install bitsandbytes before using `paged_adamw_8bit`:

```bash
uv pip install bitsandbytes
LAUNCHER=gr00t/experiment/launch_finetune_asus.py MODE=right_only \
    bash examples/Openarm_LinkerHandO6/finetune_openarm_o6.sh
```

## Open-loop evaluation

The active limbs are baked into the checkpoint at train time, so eval needs **no** modality
config — just the matching checkpoint. `MODE` only labels the output plot.

```bash
MODE=right_only \
CHECKPOINT_PATH=/data/.../gr00t_openarm_o6_right_only_0531/checkpoint-50000 \
    bash examples/Openarm_LinkerHandO6/openloop_eval_openarm_o6.sh
```

Knobs: `DATASET_PATH`, `TRAJ_IDS` (space-separated for multiple, e.g. `"0 1 2"`), `STEPS`,
`ACTION_HORIZON`, `DENOISING_STEPS`, `SAVE_PLOT_PATH`. When a single trajectory is evaluated
and `SAVE_PLOT_PATH` is unset, the wrapper saves under the checkpoint's `open_loop_eval/`
directory. For multiple trajectories, leave `SAVE_PLOT_PATH` unset to avoid overwriting plots,
or run one trajectory per invocation with an explicit path.

## Closed-loop / serving

Start a policy server (pass the matching preset so `NEW_EMBODIMENT` is registered):

```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path /data/.../gr00t_openarm_o6_right_only_0531/checkpoint-50000 \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/Openarm_LinkerHandO6/openarm_o6_config_right_only.py
```

## Files

| file | purpose |
|------|---------|
| [`openarm_o6_modalities.py`](openarm_o6_modalities.py) | shared `build_openarm_o6_config()` builder (no side effects) |
| [`openarm_o6_config.py`](openarm_o6_config.py) | bimanual preset (registers `NEW_EMBODIMENT`) |
| [`openarm_o6_config_right_only.py`](openarm_o6_config_right_only.py) | right arm + hand preset |
| [`finetune_openarm_o6.sh`](finetune_openarm_o6.sh) | MODE-aware fine-tuning launcher |
| [`openloop_eval_openarm_o6.sh`](openloop_eval_openarm_o6.sh) | open-loop evaluation |
