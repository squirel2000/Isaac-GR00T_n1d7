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

"""Custom Trainer with simple profiling utilities.

This subclass of HuggingFace's ``Trainer`` measures:
1. Data loading latency (time between the end of the previous ``training_step`` and
   the start of the current ``training_step``).
2. Forward-pass latency (time spent inside the base ``training_step`` implementation,
   which essentially wraps the model's forward / loss computation).

The statistics are logged via ``self.log`` every ``profile_log_interval`` steps and
also sent to the standard ``logging`` logger.  This is *not* meant to be a fully
fledged profiler – it is a quick, lightweight way to confirm whether the training
pipeline is bottlenecked by data loading or by the model's computation.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from typing import Any, Optional

import numpy as np
import torch
from transformers.trainer import TRAINER_STATE_NAME, Trainer, TrainerState, get_last_checkpoint
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction


class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()


class _BatchIterator:
    """Lightweight iterator that yields pre-collated batches."""

    def __init__(self, buffer, bs, collator, total_steps):
        self._buffer = buffer
        self._bs = bs
        self._collate = collator
        self._total_steps = total_steps
        self._produced = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self._total_steps

    def __next__(self):
        if self._produced >= self._total_steps:
            raise StopIteration

        # Fast path – single lock acquisition inside ``sample_batch``.
        batch_samples = self._buffer.sample_batch(self._bs)  # type: ignore[attr-defined]
        self._produced += 1
        return self._collate(batch_samples)


class _PrefetchIterator:
    def __init__(self, buffer, bs, collate_fn, total_steps):
        self.buffer = buffer
        self.bs = bs
        self.collate = collate_fn
        self.total = total_steps
        self.produced = 0

        self._q = queue.Queue(maxsize=4)
        self._stop = False

        # Start background worker
        self._worker = threading.Thread(target=self._fill)
        self._worker.daemon = True
        self._worker.start()

    def _fill(self):
        while not self._stop:
            if self.produced + self._q.qsize() >= self.total:
                break
            # block if queue is full
            samples = self.buffer.sample_batch(self.bs)
            batch = self.collate(samples)
            self._q.put(batch)

    def __iter__(self):
        return self

    def __len__(self):
        return self.total

    def __next__(self):
        if self.produced >= self.total:
            self._stop = True
            # in case worker is blocked on put()
            raise StopIteration
        batch = self._q.get()  # this will block until the next batch is ready
        self.produced += 1
        return batch


def _batch_accuracy(
    preds: torch.Tensor, labels: torch.Tensor, action_offset: Optional[int] = None
) -> torch.Tensor:  # noqa: D401
    """Compute token-level accuracy, ignoring ``-100`` label positions.

    Args:
        preds: Predicted token ids of shape ``(batch, seq_len)``.
        labels: Ground-truth label ids with the same shape as ``preds``.

    Returns:
        Scalar tensor with the fraction of correctly predicted labels in the
        current batch.
    """
    # casual prediction
    # Shift so that tokens < n predict n
    # https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py#L60
    preds = preds[:, :-1]
    labels = labels[:, 1:]

    # Ignore positions with label == -100 (HF convention)
    mask = labels != -100

    if action_offset is not None:
        # we offset the labels to the action tokens range, with normal tokens in the negatives
        labels = labels - action_offset

    correct = (preds == labels) & mask

    # Avoid division by zero for empty masks (should not happen in practice)
    denom = mask.sum().clamp(min=1)
    accuracy = correct.sum().float() / denom.float()
    return accuracy


# Global variables for batched evaluation metrics
_eval_accuracy_accumulated_correct = 0
_eval_accuracy_accumulated_total = 0
_per_joint_loss_sum: Optional[np.ndarray] = None
_per_joint_mask_sum: Optional[np.ndarray] = None


def _compute_per_joint_metrics(
    eval_pred: EvalPrediction, compute_result: bool = True
) -> dict[str, float]:
    """Per-dim (and aggregate) held-out action MSE from prediction_step outputs.

    With batch_eval_metrics off (our setup), HF concatenates every eval batch and
    calls this once with the full array and compute_result defaulting to True. The
    per-batch accumulation path (compute_result=False) is kept so the function also
    works unchanged if batched eval is ever re-enabled.

    predictions shape: [total_B, 2*action_dim] (or [B, 2*action_dim] per batch)
      [:, :D]  action_loss summed over the time horizon (already masked in forward())
      [:, D:]  action_mask summed over the time horizon (valid timestep count per dim)

    The MSE here is the flow-matching velocity-field MSE on held-out data — the
    same quantity minimized during training, so it is a valid overfitting signal.
    It is NOT a K-step denoise / action-space rollout error.
    """
    global _per_joint_loss_sum, _per_joint_mask_sum

    preds = eval_pred.predictions  # [B, 2*D]
    D = preds.shape[-1] // 2
    batch_loss = preds[:, :D].sum(axis=0)  # [D]
    batch_mask = preds[:, D:].sum(axis=0)  # [D]

    if _per_joint_loss_sum is None:
        _per_joint_loss_sum = batch_loss
        _per_joint_mask_sum = batch_mask
    else:
        _per_joint_loss_sum = _per_joint_loss_sum + batch_loss
        _per_joint_mask_sum = _per_joint_mask_sum + batch_mask

    if compute_result:
        per_joint = _per_joint_loss_sum / np.maximum(_per_joint_mask_sum, 1e-6)
        result = {
            f"action_dim_{i:03d}_mse": float(v)
            for i, (v, m) in enumerate(zip(per_joint, _per_joint_mask_sum))
            if m > 0
        }
        # Aggregate, mask-weighted MSE across all dims — a single curve for
        # overfitting / best-step detection (use as save_best_eval_metric_name).
        total_loss = float(_per_joint_loss_sum.sum())
        total_mask = float(_per_joint_mask_sum.sum())
        result["action_mse"] = total_loss / max(total_mask, 1e-6)
        _per_joint_loss_sum = None
        _per_joint_mask_sum = None
        return result
    else:
        return {}


def compute_eval_accuracy(
    eval_pred: EvalPrediction, compute_result: bool, action_offset: Optional[int] = None
):
    logits = eval_pred.predictions[0]
    if action_offset is not None:
        logits = logits[..., action_offset:]
    preds = logits.argmax(axis=-1)
    labels = eval_pred.label_ids

    preds = preds[:, :-1]
    labels = labels[:, 1:]

    # Ignore positions with label == -100 (HF convention)
    mask = labels != -100

    if action_offset is not None:
        # we offset the labels to the action tokens range, with normal tokens in the negatives
        labels = labels - action_offset

    correct = ((preds == labels) & mask).sum()
    total = mask.sum()

    global _eval_accuracy_accumulated_correct, _eval_accuracy_accumulated_total
    _eval_accuracy_accumulated_correct += correct
    _eval_accuracy_accumulated_total += total

    if compute_result:
        accuracy = _eval_accuracy_accumulated_correct / max(_eval_accuracy_accumulated_total, 1)
        _eval_accuracy_accumulated_correct = 0
        _eval_accuracy_accumulated_total = 0
        return {"eval_accuracy": accuracy}
    else:
        return {}


class Gr00tTrainer(Trainer):
    """Trainer that bypasses torch dataloader and makes data collator async."""

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:  # noqa: D401 – simple description above
        """Initialize the trainer.

        Args:
            *args: Positional arguments forwarded to ``Trainer``.
        """
        self.action_offset = kwargs.pop("action_offset", None)
        self.multiprocessing_context = kwargs.pop("multiprocessing_context", "fork")
        # Per-eval-batch counter for deterministic eval seeding (see prediction_step).
        self._eval_rng_counter = 0
        super().__init__(
            *args,
            **kwargs,
            compute_metrics=_compute_per_joint_metrics,
            # compute_metrics=partial(compute_eval_accuracy, action_offset=self.action_offset),
        )

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        # Hide epoch from logged metrics as it's misleading for Iterable datasets.
        epoch = self.state.epoch
        self.state.epoch = None
        super().log(logs, start_time=start_time)
        self.state.epoch = epoch

    def get_train_dataloader(self):  # noqa: D401
        """Return a iterable dataloader without skipping the data during resume, but reseed the dataset instead."""

        # Fall back to default behaviour if not using the custom buffer.
        # During resume, don't skip the data
        self.args.ignore_data_skip = True
        curr_global_step = self.state.global_step
        print(f"Current global step: {curr_global_step}")
        if curr_global_step > 0:
            # ``new_seed`` MUST be the same on every rank: ``ShardedMixtureDataset``
            # builds its shard schedule from this seed and partitions disjointly
            # by index, so a per-rank delta here would cause sample duplication
            # / loss across ranks. Both inputs are rank-symmetric (the dataset's
            # own seed was set rank-symmetrically at __init__, and global_step
            # is read from TrainerState which is broadcast via rendezvous).
            new_seed = self.train_dataset.seed + curr_global_step
            self.train_dataset.reset_seed(new_seed)
            print(
                f"Resetting seed to {new_seed}. Please note that this will make the experiment non-reproducible."
            )

        print("Creating custom train dataloader")
        # Handle the case where the dataset is an IterableDataset
        data_collator = self.data_collator
        data_collator = self._get_collator_with_removed_columns(
            data_collator, description="training"
        )
        # Use persistent workers for sharded dataset if num_workers is greater than 0
        persistent_workers = self.args.dataloader_num_workers > 0

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": persistent_workers,
        }

        # multiprocessing_context can only be used with num_workers > 0
        if self.args.dataloader_num_workers > 0:
            dataloader_params["multiprocessing_context"] = self.multiprocessing_context

        return torch.utils.data.DataLoader(self.train_dataset, **dataloader_params)

    def get_eval_dataloader(self, eval_dataset=None):
        """Return an eval dataloader built the same way as the train dataloader.

        The default HF Trainer eval dataloader is wrapped by accelerate's
        DataLoaderDispatcher, which tries to NCCL-broadcast CPU tensors across
        ranks and fails for our IterableDataset. A plain DataLoader (matching
        get_train_dataloader) bypasses that.
        """
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # Reset the per-batch RNG counter so deterministic eval seeding (see
        # prediction_step) restarts identically on every eval run.
        self._eval_rng_counter = 0

        # Multi-GPU caveat: this plain DataLoader is intentionally NOT accelerate-
        # wrapped, so each rank reads its own eval shards. Shard sizes differ, so
        # per-rank eval-batch counts differ, and the per-batch metric all-gather can
        # deadlock when ranks make an unequal number of collective calls. Single-GPU
        # eval is unaffected; warn loudly otherwise.
        if self.args.world_size > 1:
            logging.warning(
                "Held-out eval uses an unsharded plain DataLoader per rank; with "
                "world_size=%d, unequal per-rank eval-batch counts may hang the metric "
                "all-gather. Run eval on a single process if it stalls.",
                self.args.world_size,
            )

        data_collator = self.data_collator
        data_collator = self._get_collator_with_removed_columns(
            data_collator, description="evaluation"
        )
        persistent_workers = self.args.dataloader_num_workers > 0

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": persistent_workers,
        }

        if self.args.dataloader_num_workers > 0:
            dataloader_params["multiprocessing_context"] = self.multiprocessing_context

        return torch.utils.data.DataLoader(eval_dataset, **dataloader_params)

    def train(
        self,
        resume_from_checkpoint=None,
        **kwargs,
    ):
        """Correctly set self.state from checkpoint so get_train_dataloader can read from it."""
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                logging.warning(
                    f"No valid checkpoint found in output directory ({self.args.output_dir})"
                )

        if resume_from_checkpoint is not None:
            logging.info(f"Resuming from checkpoint {resume_from_checkpoint}")
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )

        return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

    # ------------------------------------------------------------------
    # Loss / accuracy computation override
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):  # type: ignore[override]
        """Compute loss *and* log token-level accuracy every training step.

        We delegate the heavy-lifting (including label smoothing, custom loss
        functions, etc.) to the parent ``Trainer.compute_loss`` implementation
        by calling it with ``return_outputs=True``.  After obtaining the loss
        *and* model outputs, we calculate accuracy and push it to the logger.
        """

        # Use parent implementation to preserve built-in functionality.
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        # import ipdb; ipdb.set_trace()
        # # save the model's embedding for the first step
        # input_embeddings = model.get_input_embeddings().weight.data.cpu()
        # output_embeddings = model.get_output_embeddings().weight.data.cpu()
        # torch.save(input_embeddings, f"input_embeddings_{self.state.global_step}.pt")
        # torch.save(output_embeddings, f"output_embeddings_{self.state.global_step}.pt")

        # Record last loss for testing purposes.
        self.loss = loss

        # --------------------------------------------------------------
        # Accuracy calculation
        # --------------------------------------------------------------
        if (
            self.state.global_step % self.args.logging_steps == 0
            and model.training
            and "labels" in inputs
        ):
            if self.action_offset is not None:
                preds = outputs.logits.detach()[:, :, self.action_offset :].argmax(dim=-1).cpu()
            else:
                preds = outputs.logits.detach().argmax(dim=-1).cpu()
            with torch.no_grad():
                acc_local = _batch_accuracy(
                    preds, inputs["labels"].to(device=preds.device), self.action_offset
                )
            acc_tensor = torch.tensor(acc_local.item(), device=loss.device)
            acc_mean = self._nested_gather(acc_tensor).mean().item()

            if self.args.local_rank in (-1, 0):
                self.log({"train_accuracy": acc_mean})

                # Log a sample of ground-truth vs predicted action tokens from
                # the first batch element so users can verify the model is
                # learning the right behaviors.
                shifted_labels = inputs["labels"][:1, 1:].cpu()
                shifted_preds = preds[:1, :-1]
                mask_0 = shifted_labels[0] != -100
                gt_tokens = shifted_labels[0][mask_0][:20]
                if self.action_offset is not None:
                    gt_tokens = gt_tokens - self.action_offset
                gt_sample = gt_tokens.tolist()
                pred_sample = shifted_preds[0][mask_0[: shifted_preds.shape[1]]][:20].tolist()
                logging.info(
                    "Step %d — GT vs Pred (first 20 action tokens, batch[0]):\n"
                    "  GT:   %s\n  Pred: %s",
                    self.state.global_step,
                    gt_sample,
                    pred_sample,
                )

        return (loss, outputs) if return_outputs else loss

    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        """Return per-dim action loss/mask sums for compute_metrics aggregation.

        logits shape: [B, 2*action_dim]
          [:, :D]  action_loss summed over time (already masked in forward())
          [:, D:]  action_mask summed over time
        HF Trainer concatenates across eval batches → [total_B, 2*D].
        """
        inputs = self._prepare_inputs(inputs)
        # Deterministic eval (common random numbers): forward() draws a fresh
        # flow-matching timestep + noise on every call, so without fixing the RNG the
        # held-out MSE would jitter from sampling noise rather than from the model
        # improving across eval steps. Seed per eval batch (counter reset at the start
        # of each eval in get_eval_dataloader) and restore the global RNG afterwards so
        # training reproducibility is untouched. Identical eval-data order across runs
        # plus the same per-batch seed ⇒ the same batch sees the same noise every eval.
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(self.args.seed + self._eval_rng_counter)
        self._eval_rng_counter += 1
        try:
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        finally:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
        loss = loss.detach()

        if prediction_loss_only:
            return loss, None, None

        action_loss = outputs["action_loss"].detach().float()  # [B, T, D]
        action_mask = outputs["action_mask"].detach().float()  # [B, T, D]
        loss_sum = action_loss.sum(dim=1)  # [B, D]
        mask_sum = action_mask.sum(dim=1)  # [B, D]
        logits = torch.cat([loss_sum, mask_sum], dim=-1)  # [B, 2*D]
        # HF's eval loop only invokes compute_metrics when label_ids is not None
        # (both the batched and the concatenate-once paths gate on it), so return
        # a placeholder label. The real per-joint MSE is carried entirely in logits.
        dummy_labels = torch.zeros(logits.shape[0], 1, device=logits.device)
        return loss, logits, dummy_labels
