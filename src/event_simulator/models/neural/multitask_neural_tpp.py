from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..common.conditions import CONDITION_NAMES, condition_feature_vector, condition_flags
from ..common.datasets import LearnedTPPBaseline, SequenceDataset
from ..common.networks import GRUMultitaskTPPModel
from ..common.labels import event_label
from ..common.replay import ReplayState


def actual_state_until_time(
    base_state: ReplayState,
    future_events: list[dict[str, Any]],
    summary: dict[str, Any],
    target_time: float,
) -> ReplayState:
    rolled_state = base_state.clone()
    for event in future_events:
        if float(event["time_s"]) > target_time:
            break
        rolled_state.update(event, summary)
    rolled_state.current_time = max(rolled_state.current_time, target_time)
    return rolled_state
class MultitaskNeuralTPPBaseline(LearnedTPPBaseline):
    name = "multitask_neural_tpp"
    description = "GRU event predictor with direct fixed-horizon condition heads for congestion and corridor-pressure forecasting"

    def __init__(
        self,
        context_len: int = 32,
        epochs: int = 16,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        device: str = "auto",
        condition_horizons: tuple[float, ...] = (10.0, 30.0, 60.0),
        condition_loss_weight: float = 0.15,
    ) -> None:
        self.context_len = context_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.condition_horizons = condition_horizons
        self.condition_loss_weight = condition_loss_weight
        self.condition_thresholds = [0.5] * (len(self.condition_horizons) * len(CONDITION_NAMES))
        self.condition_scalers = [(1.0, 0.0)] * (len(self.condition_horizons) * len(CONDITION_NAMES))

    def build_multitask_samples(self, train_runs: list[dict[str, Any]]) -> list[dict[str, object]]:
        samples: list[dict[str, object]] = []
        for run_index, run in enumerate(train_runs):
            state = ReplayState()
            events = run["events"]
            duration = float(run["summary"]["duration_seconds"])
            for event_index, (current_event, next_event) in enumerate(zip(events, events[1:])):
                state.update(current_event, run["summary"])
                history_labels = state.context_labels[-self.context_len :]
                history_deltas = state.context_deltas[-self.context_len :]
                target_delta = max(0.01, float(next_event["time_s"]) - state.current_time)
                future_events = events[event_index + 1 :]
                condition_targets: list[float] = []
                condition_mask: list[float] = []
                for horizon in self.condition_horizons:
                    target_time = state.current_time + horizon
                    valid = target_time <= duration
                    if valid:
                        target_state = actual_state_until_time(state, future_events, run["summary"], target_time)
                        flags = condition_flags(target_state)
                    else:
                        flags = {name: False for name in CONDITION_NAMES}
                    condition_targets.extend(float(flags[name]) for name in CONDITION_NAMES)
                    condition_mask.extend([1.0 if valid else 0.0] * len(CONDITION_NAMES))
                samples.append(
                    {
                        "run_id": run_index,
                        "labels": torch.tensor([self.label_to_id[label] for label in history_labels], dtype=torch.long),
                        "deltas": torch.tensor(history_deltas, dtype=torch.float32),
                        "state_features": torch.tensor(condition_feature_vector(state, run["summary"]), dtype=torch.float32),
                        "target_label": self.label_to_id[event_label(next_event)],
                        "target_delta": math.log1p(target_delta),
                        "condition_targets": torch.tensor(condition_targets, dtype=torch.float32),
                        "condition_mask": torch.tensor(condition_mask, dtype=torch.float32),
                    }
                )
        return samples

    def collate(self, batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        collated = self.collate_sequence_batch(batch)
        collated["condition_targets"] = torch.stack([item["condition_targets"] for item in batch])
        collated["condition_mask"] = torch.stack([item["condition_mask"] for item in batch])
        return collated

    def fit(self, train_runs: list[dict[str, Any]]) -> None:
        torch.manual_seed(19)
        random.seed(19)
        self.runtime_device = self.resolve_device()
        self.initialize_label_vocab(train_runs)
        samples = self.build_multitask_samples(train_runs)
        loader = DataLoader(SequenceDataset(samples), batch_size=self.batch_size, shuffle=True, collate_fn=self.collate)
        threshold_loader = DataLoader(
            SequenceDataset(samples),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate,
        )
        self.model = GRUMultitaskTPPModel(
            vocab_size=len(self.label_to_id) + 1,
            state_dim=len(samples[0]["state_features"]),
            hidden_dim=72,
            embedding_dim=24,
            condition_dim=len(self.condition_horizons) * len(CONDITION_NAMES),
        ).to(self.runtime_device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        label_loss_fn = nn.CrossEntropyLoss()
        delta_loss_fn = nn.SmoothL1Loss()
        pos_counts = torch.stack([sample["condition_targets"] for sample in samples]).sum(dim=0)
        valid_counts = torch.stack([sample["condition_mask"] for sample in samples]).sum(dim=0).clamp(min=1.0)
        neg_counts = (valid_counts - pos_counts).clamp(min=0.0)
        pos_weight = (neg_counts / pos_counts.clamp(min=1.0)).clamp(min=0.5, max=8.0).to(self.runtime_device)
        condition_loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        self.model.train()
        self.training_log(
            f"starting fit epochs={self.epochs} batches_per_epoch={len(loader)} device={self.runtime_device}"
        )
        for epoch_index in range(self.epochs):
            epoch_start = self.training_epoch_start()
            epoch_loss_sum = 0.0
            batch_count = 0
            for batch in loader:
                batch = self.move_batch_to_device(batch, self.runtime_device)
                optimizer.zero_grad()
                logits, log_delta, condition_logits = self.model(
                    batch["labels"], batch["deltas"], batch["lengths"], batch["state_features"]
                )
                loss = label_loss_fn(logits, batch["target_label"]) + 0.35 * delta_loss_fn(log_delta, batch["target_delta"])
                condition_loss = condition_loss_fn(condition_logits, batch["condition_targets"])
                masked_condition_loss = (condition_loss * batch["condition_mask"]).sum() / batch["condition_mask"].sum().clamp(min=1.0)
                loss = loss + self.condition_loss_weight * masked_condition_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss_sum += float(loss.detach().item())
                batch_count += 1
            self.training_epoch_end(
                epoch_index + 1,
                self.epochs,
                epoch_start,
                epoch_loss_sum / max(1, batch_count),
            )
        self.model.eval()
        self.condition_scalers, self.condition_thresholds = self.tune_condition_calibration(threshold_loader)

    def tune_condition_calibration(self, loader: DataLoader) -> tuple[list[tuple[float, float]], list[float]]:
        logits_by_dim: list[list[float]] = [[] for _ in range(len(self.condition_thresholds))]
        targets_by_dim: list[list[float]] = [[] for _ in range(len(self.condition_thresholds))]
        with torch.no_grad():
            for batch in loader:
                batch = self.move_batch_to_device(batch, self.runtime_device)
                _, _, condition_logits = self.model(batch["labels"], batch["deltas"], batch["lengths"], batch["state_features"])
                logits = condition_logits.cpu()
                targets = batch["condition_targets"].cpu()
                mask = batch["condition_mask"].cpu()
                for dim in range(logits.size(1)):
                    valid = mask[:, dim] > 0
                    if valid.any():
                        logits_by_dim[dim].extend(logits[valid, dim].tolist())
                        targets_by_dim[dim].extend(targets[valid, dim].tolist())
        scalers = self.fit_platt_scalers(logits_by_dim, targets_by_dim)
        probs_by_dim = []
        for dim, logits in enumerate(logits_by_dim):
            scale, bias = scalers[dim]
            probs_by_dim.append([float(torch.sigmoid(torch.tensor(value * scale + bias)).item()) for value in logits])
        thresholds = self.tune_condition_thresholds_from_probs(probs_by_dim, targets_by_dim)
        return scalers, thresholds

    def predict(self, state: ReplayState, summary: dict[str, Any]) -> tuple[str, float]:
        labels = state.context_labels[-self.context_len :]
        deltas = state.context_deltas[-self.context_len :]
        with torch.no_grad():
            logits, log_delta, _ = self.model(
                torch.tensor([[self.label_to_id.get(label, 0) for label in labels]], dtype=torch.long, device=self.runtime_device),
                torch.tensor([deltas], dtype=torch.float32, device=self.runtime_device),
                torch.tensor([len(labels)], dtype=torch.long, device=self.runtime_device),
                torch.tensor([condition_feature_vector(state, summary)], dtype=torch.float32, device=self.runtime_device),
            )
        predicted_id = int(torch.argmax(logits, dim=-1).item())
        return self.id_to_label.get(predicted_id, "phase_change:NS_GREEN"), state.current_time + max(0.01, math.expm1(float(log_delta.item())))

    def predict_time_conditions(
        self,
        state: ReplayState,
        summary: dict[str, Any],
        horizons: list[float],
    ) -> dict[str, dict[str, bool]] | None:
        requested = {f"{int(horizon)}s" for horizon in horizons}
        labels = state.context_labels[-self.context_len :]
        deltas = state.context_deltas[-self.context_len :]
        with torch.no_grad():
            _, _, condition_logits = self.model(
                torch.tensor([[self.label_to_id.get(label, 0) for label in labels]], dtype=torch.long, device=self.runtime_device),
                torch.tensor([deltas], dtype=torch.float32, device=self.runtime_device),
                torch.tensor([len(labels)], dtype=torch.long, device=self.runtime_device),
                torch.tensor([condition_feature_vector(state, summary)], dtype=torch.float32, device=self.runtime_device),
            )
        condition_logits = self.apply_platt_scalers(condition_logits, self.condition_scalers)
        condition_probs = torch.sigmoid(condition_logits[0]).tolist()
        output: dict[str, dict[str, bool]] = {}
        for horizon_index, horizon in enumerate(self.condition_horizons):
            horizon_key = f"{int(horizon)}s"
            start = horizon_index * len(CONDITION_NAMES)
            output[horizon_key] = {
                name: condition_probs[start + name_index] >= self.condition_thresholds[start + name_index]
                for name_index, name in enumerate(CONDITION_NAMES)
            }
        return {key: value for key, value in output.items() if key in requested}

    def predict_time_condition_scores(
        self,
        state: ReplayState,
        summary: dict[str, Any],
        horizons: list[float],
    ) -> dict[str, dict[str, float]] | None:
        requested = {f"{int(horizon)}s" for horizon in horizons}
        labels = state.context_labels[-self.context_len :]
        deltas = state.context_deltas[-self.context_len :]
        with torch.no_grad():
            _, _, condition_logits = self.model(
                torch.tensor([[self.label_to_id.get(label, 0) for label in labels]], dtype=torch.long, device=self.runtime_device),
                torch.tensor([deltas], dtype=torch.float32, device=self.runtime_device),
                torch.tensor([len(labels)], dtype=torch.long, device=self.runtime_device),
                torch.tensor([condition_feature_vector(state, summary)], dtype=torch.float32, device=self.runtime_device),
            )
        condition_logits = self.apply_platt_scalers(condition_logits, self.condition_scalers)
        condition_probs = torch.sigmoid(condition_logits[0]).tolist()
        output: dict[str, dict[str, float]] = {}
        for horizon_index, horizon in enumerate(self.condition_horizons):
            horizon_key = f"{int(horizon)}s"
            start = horizon_index * len(CONDITION_NAMES)
            output[horizon_key] = {
                name: float(condition_probs[start + name_index])
                for name_index, name in enumerate(CONDITION_NAMES)
            }
        return {key: value for key, value in output.items() if key in requested}

    def save_checkpoint(self, path: str | Path) -> None:
        payload = self.checkpoint_base_payload()
        payload["model_state_dict"] = self.model.state_dict()
        payload["state_dim"] = int(self.model.shared[0].in_features - self.model.gru.hidden_size)
        payload["hidden_dim"] = int(self.model.gru.hidden_size)
        payload["embedding_dim"] = int(self.model.embedding.embedding_dim)
        self.save_torch_checkpoint(path, payload)

    @classmethod
    def load_checkpoint(cls, path: str | Path, device: str = "auto") -> "MultitaskNeuralTPPBaseline":
        payload = torch.load(path, map_location="cpu")
        model = cls(
            context_len=int(payload["context_len"]),
            epochs=int(payload.get("epochs") or 16),
            batch_size=int(payload.get("batch_size") or 64),
            learning_rate=float(payload.get("learning_rate") or 1e-3),
            device=device if device != "auto" else str(payload.get("runtime_device", "cpu")),
            condition_horizons=tuple(float(value) for value in payload.get("condition_horizons", (10.0, 30.0, 60.0))),
            condition_loss_weight=float(payload.get("condition_loss_weight") or 0.15),
        )
        model.runtime_device = model.resolve_device()
        model.restore_base_state(payload)
        model.model = GRUMultitaskTPPModel(
            vocab_size=len(model.label_to_id) + 1,
            state_dim=int(payload["state_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            embedding_dim=int(payload["embedding_dim"]),
            condition_dim=len(model.condition_horizons) * len(CONDITION_NAMES),
        ).to(model.runtime_device)
        model.model.load_state_dict(payload["model_state_dict"])
        model.model.eval()
        return model
