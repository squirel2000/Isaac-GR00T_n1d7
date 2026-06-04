# GR00T N1.7 fine-tuning on a single RTX 5090 (32 GB)

End-to-end recipe verified on this box (RTX 5090 D 32 GB · Blackwell **sm_120** · driver 580 · CUDA 13 · Ubuntu 22.04 · Python 3.10). Full 50 000-step OpenArm-O6 fine-tune completed successfully with the configuration below.

## 1. Setup (once)

```bash
cd /data/Gits/IsaacLab-GR00T/Isaac-GR00T_n1d7
uv sync --python 3.10                       # installs PyTorch 2.7.1+cu128 (ships sm_120 kernels)
uv pip install bitsandbytes                 # for OPTIM=paged_adamw_8bit (sm_120 verified)
uv run hf auth login                        # backbone nvidia/Cosmos-Reason2-2B is GATED
sudo apt install -y ffmpeg                  # torchcodec video decoder
```

**Stability fix — REQUIRED before any long run** (RTX 5090 transient power spikes hard-reboot the box; the previously-working `-lgc 1800,2200` failed at ~step 5754, so cap further):

```bash
sudo tee /etc/systemd/system/nv-powercap.service >/dev/null <<'EOF'
[Unit]
Description=Cap NVIDIA RTX 5090 clocks/power to prevent PSU-transient reboots
After=multi-user.target
[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi -pm 1
ExecStart=/usr/bin/nvidia-smi -lgc 510,1500
ExecStart=/usr/bin/nvidia-smi -pl 400
RemainAfterExit=yes
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now nv-powercap.service
```
Under load: SM clock pins ~1275 MHz, ~290 W. Runtime `nvidia-smi -pm/-lgc/-pl` reset every reboot — the systemd unit re-applies them automatically. Removal: see §6.

## 2. TL;DR — verified working 50k command

`paged_adamw_8bit` + per-device batch 16 × grad-accum 2 (effective 32), workers 0, auto-resume from latest checkpoint on any interruption. This *exact* command completed 50 000 steps on the 1000-episode OpenArm-O6 dataset:

```bash
cd /data/Gits/IsaacLab-GR00T/Isaac-GR00T_n1d7
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPTIM=paged_adamw_8bit
nohup uv run python gr00t/experiment/launch_finetune_asus.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /data/Gits/IsaacLab-GR00T/datasets/OpenArm_O6_CanSorting_dataset_0408 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/Openarm_LinkerHandO6/openarm_o6_config.py \
  --num-gpus 1 --output-dir /data/Gits/IsaacLab-GR00T/artifacts/gr00t_openarm_O6_CanSorting_0408_50k \
  --max-steps 50000 --save-steps 2000 --save-total-limit 3 \
  --global-batch-size 16 --gradient-accumulation-steps 2 --dataloader-num-workers 0 \
  --use-wandb --wandb-project gr00t-openarm-o6_50k \
  > /data/Gits/IsaacLab-GR00T/artifacts/gr00t_openarm_O6_CanSorting_0408_50k.log 2>&1 &
```

**Single-arm (right_only) variant** — same dataset, but only the right arm + right hand are used.
Swap the modality config (and pick a distinct output dir); everything else is identical:

```bash
  --modality-config-path examples/Openarm_LinkerHandO6/openarm_o6_config_right_only.py \
  --output-dir /data/Gits/IsaacLab-GR00T/artifacts/gr00t_openarm_O6_CanSorting_0408_right_only_50k \
```

VRAM is the same as bimanual (the backbone dominates; only the 26→13 action/state dims change),
so the §2 batch/optimizer settings carry over unchanged.

## 3. Why the defaults differ from upstream

| Knob | Upstream | Here | Reason |
|---|---|---|---|
| Optimizer | `adamw_torch` | **`paged_adamw_8bit`** (or `adafactor`) | AdamW needs ~13 GB fp32 momentum for 1.62 B trainable params → OOM on 32 GB at the first `optimizer.step()`. 8-bit Adam ≈ 3 GB state (peak ~27.5 GB); adafactor ≈ 10 MB (peak ~25 GB). |
| Batch | `--global-batch-size 32` | **16 × grad-accum 2** | Effective 32; per-device 16 fits comfortably. |
| Workers | 4 | **0** | `multiprocessing_context="fork"` + video decode → mutex deadlock mid-run. 0 workers = no fork = no deadlock. |
| Launcher | `gr00t/experiment/launch_finetune.py` | **`gr00t/experiment/launch_finetune_asus.py`** | Upstream hard-codes the optimizer; the wrapper overrides it via env (`OPTIM`, `GRAD_CKPT`) without touching repo code. |
| GPU power | default 600 W, unlocked clocks | **lgc 510,1500 + pl 400** | See §1 — required for stability. |

## 4. Fine-tuning a NEW dataset → `--modality-config-path`

Two files; don't confuse them:

- **`<dataset>/meta/modality.json`** (data side) — splits the flat state/action vectors into named field groups. Usually shipped with the dataset.
- **`<your>.py` passed via `--modality-config-path`** — registers a `ModalityConfig` (which keys, prediction horizon, action representation) for `EmbodimentTag.NEW_EMBODIMENT`.

Procedure: read `meta/modality.json`, build one `ModalityConfig` per block (`video`/`state`/`action`/`language`) whose `modality_keys` match exactly; give `action` one `ActionConfig` per key (`rep`, `type`, `format`); then `register_modality_config(cfg, EmbodimentTag.NEW_EMBODIMENT)`.

**Action representation** (data may be absolute; processor auto-converts on the fly):

| `rep` | What the processor does | Use for |
|---|---|---|
| `ABSOLUTE` | no conversion (absolute target) | grippers / dex-hand fingers (binary-like) |
| `RELATIVE` | NON_EEF: `action − current_state[-1]`; EEF: SE(3) pose composition | arms — N1.7's pretrained prior |
| `DELTA` | **unimplemented in N1.7 — do not use** | — |

Worked example: bimanual **OpenArm-O6 CanSorting** (26-D state/action, 1 RGB cam, 1000 eps) →
[`examples/Openarm_LinkerHandO6/openarm_o6_config.py`](examples/Openarm_LinkerHandO6/openarm_o6_config.py)
(arms RELATIVE, hands ABSOLUTE, horizon 16; validated: 417 shards / 426 315 samples, no key errors).

## 5. Monitoring a run

```bash
tail -f <logfile>                                        # tqdm progress bar
watch -n2 nvidia-smi                                     # VRAM / util / power / clocks
# live loss/grad-norm/lr (stdout {'loss':...} lines are buffered):
python -c "import json; d=json.load(open('<output-dir>/checkpoint-<N>/trainer_state.json')); \
[print(x['step'], x['loss']) for x in d['log_history'] if 'loss' in x][-10:]"
```

**Weights & Biases:** `uv run wandb login` (once), then add `--use-wandb [--wandb-project NAME]` to the launch command (already on the §2 example). Air-gapped: `WANDB_MODE=offline` + later `wandb sync`.

## 6. Stability fix — RTX 5090 hard reboots (REQUIRED)

**Symptom.** Box hard-reboots minutes into long runs; `last` shows session ended in `crash`; kernel log just stops (no panic / Xid / thermal / MCE); power back ~80 s later. **Instant power loss → PSU tripping OCP/OPP on RTX 5090 transient spikes** (1.5–2× the sustained draw, briefly >1 kW). PSU here is 1200 W — watts are adequate; the problem is transient response.

**Observed escalation on this box:** no cap → reboot @1700–2500 steps · `lgc 1800,2200 + pl 400` → reboot @~5754 · **`lgc 510,1500 + pl 400` (§1) → 50 000 steps completed successfully**. Key lever is the **clock lock**, not the power cap (a power cap still allows boost spikes; locking max clock holds voltage/current down).

**Removing the cap later:**
```bash
sudo systemctl disable --now nv-powercap.service
sudo rm /etc/systemd/system/nv-powercap.service
sudo systemctl daemon-reload
sudo nvidia-smi -rgc && sudo nvidia-smi -pl 600   # un-set live settings without reboot
```

**If it still reboots at `lgc 510,1500 + pl 400` → hardware**, not software:
- Reseat the **12V-2x6 (12VHPWR)** connector at *both* ends; use the native cable (no pigtail). Contact resistance degrading over time fits the "previously-working cap stopped working" pattern.
- PSU: prefer ATX 3.1 / PCIe 5.1 with proper transient handling; single high-current rail for the GPU.

## 7. FAQ

- **GPU at 1 % util, VRAM full, no progress:** fork DataLoader deadlock. Kill the process and re-run with `--dataloader-num-workers 0`; auto-resumes from the latest checkpoint.
- **Worktree → main clone move:** nothing is trapped — `.venv` is git-ignored (`uv sync` in the main clone is fast, cache is warm); checkpoints live under `/data/...` already; deliverable scripts are in the repo root. Optionally remove the agent worktree: `git worktree remove .claude/worktrees/<name> --force`.
