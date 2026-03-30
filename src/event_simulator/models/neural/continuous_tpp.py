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
from ..common.labels import event_label
from ..common.networks import ContinuousLSTMNextEventModel
from ..common.replay import ReplayState, state_feature_vector


class ContinuousTPPBaseline(LearnedTPPBaseline):
    name = "continuous_tpp"
    description = "Continuous-time LSTM next-event predictor with elapsed-time decay and direct fixed-horizon condition heads"

    def __init__(
        self,
        context_len: int = 32,
        epochs: int = 16,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        device: str = "auto",
        condition_horizons: tuple[float, ...] = (10.0, 30.0, 60.0),
        condition_loss_weight: float = 0.12,
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
        self.validation_fraction = 0.15

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
                        target_state = state.clone()
                        for event in future_events:
                            if float(event["time_s"]) > target_time:
                                break
                            target_state.update(event, run["summary"])
                        target_state.current_time = max(target_state.current_time, target_time)
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

    def fit(self, train_runs: list[dict[str, Any]]) -> None:
        torch.manual_seed(11)
        random.seed(11)
        self.runtime_device = self.resolve_device()
        self.initialize_label_vocab(train_runs)
        samples = self.build_multitask_samples(train_runs)
        train_samples, validation_samples = self.split_train_validation_samples(
            samples,
            validation_fraction=self.validation_fraction,
            seed=11,
            group_key="run_id",
        )
        loader = DataLoader(SequenceDataset(train_samples), batch_size=self.batch_size, shuffle=True, collate_fn=self.collate)
        validation_loader = DataLoader(
            SequenceDataset(validation_samples),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate,
        )
        self.model = ContinuousLSTMNextEventModel(
            vocab_size=len(self.label_to_id) + 1,
            state_dim=len(samples[0]["state_features"]),
            hidden_dim=72,
            embedding_dim=24,
        ).to(self.runtime_device)
        self.condition_head = nn.Sequential(
            nn.Linear(72, 72),
            nn.ReLU(),
            nn.Linear(72, len(self.condition_horizons) * len(CONDITION_NAMES)),
        ).to(self.runtime_device)
        optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.condition_head.parameters()), lr=self.learning_rate)
        label_loss_fn = nn.CrossEntropyLoss()
        delta_loss_fn = nn.SmoothL1Loss()
        pos_counts = torch.stack([sample["condition_targets"] for sample in train_samples]).sum(dim=0)
        valid_counts = torch.stack([sample["condition_mask"] for sample in train_samples]).sum(dim=0).clamp(min=1.0)
        neg_counts = (valid_counts - pos_counts).clamp(min=0.0)
        pos_weight = (neg_counts / pos_counts.clamp(min=1.0)).clamp(min=0.5, max=8.0).to(self.runtime_device)
        condition_loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        self.model.train()
        self.condition_head.train()
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
                logits, log_delta = self.model(batch["labels"], batch["deltas"], batch["lengths"], batch["state_features"])
                condition_logits = self.condition_head(self._trunk(batch))
                loss = label_loss_fn(logits, batch["target_label"]) + 0.4 * delta_loss_fn(log_delta, batch["target_delta"])
                condition_loss = condition_loss_fn(condition_logits, batch["condition_targets"])
                masked_condition_loss = (condition_loss * batch["condition_mask"]).sum() / batch["condition_mask"].sum().clamp(min=1.0)
                loss = loss + self.condition_loss_weight * masked_condition_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.condition_head.parameters()), max_norm=1.0)
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
        self.condition_head.eval()
        self.condition_scalers, self.condition_thresholds = self.tune_condition_calibration(validation_loader)

    def _trunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len = batch["labels"].shape
        hidden_dim = self.model.label_head.in_features
        h = torch.zeros(batch_size, hidden_dim, device=batch["labels"].device)
        c = torch.zeros(batch_size, hidden_dim, device=batch["labels"].device)
        valid = torch.arange(seq_len, device=batch["labels"].device).unsqueeze(0) < batch["lengths"].unsqueeze(1)
        for step in range(seq_len):
            delta_step = batch["deltas"][:, step : step + 1]
            delta_log = torch.log1p(delta_step)
            embed = self.model.embedding(batch["labels"][:, step])
            h = h * self.model.decay(delta_log)
            projected = self.model.input_proj(torch.cat([embed, delta_log, valid[:, step : step + 1].float()], dim=-1))
            new_h, new_c = self.model.cell(projected, (h, c))
            mask = valid[:, step].unsqueeze(-1)
            h = torch.where(mask, new_h, h)
            c = torch.where(mask, new_c, c)
        return self.model.shared(torch.cat([h, batch["state_features"]], dim=-1))

    def tune_condition_calibration(self, loader: DataLoader) -> tuple[list[tuple[float, float]], list[float]]:
        logits_by_dim = [[] for _ in range(len(self.condition_thresholds))]
        targets_by_dim = [[] for _ in range(len(self.condition_thresholds))]
        with torch.no_grad():
            for batch in loader:
                batch = self.move_batch_to_device(batch, self.runtime_device)
                condition_logits = self.condition_head(self._trunk(batch))
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
        thresholds = self.tune_condition_thresholds_from_probs(
            probs_by_dim,
            targets_by_dim,
            candidate_thresholds=[round(step * 0.05, 2) for step in range(2, 19)],
        )
        return scalers, thresholds

    def collate(self, batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        collated = self.collate_sequence_batch(batch)
        collated["condition_targets"] = torch.stack([item["condition_targets"] for item in batch])
        collated["condition_mask"] = torch.stack([item["condition_mask"] for item in batch])
        return collated

    def predict(self, state: ReplayState, summary: dict[str, Any]) -> tuple[str, float]:
        labels = state.context_labels[-self.context_len :]
        deltas = state.context_deltas[-self.context_len :]
        with torch.no_grad():
            logits, log_delta = self.model(
                torch.tensor([[self.label_to_id.get(label, 0) for label in labels]], dtype=torch.long, device=self.runtime_device),
                torch.tensor([deltas], dtype=torch.float32, device=self.runtime_device),
                torch.tensor([len(labels)], dtype=torch.long, device=self.runtime_device),
                torch.tensor([condition_feature_vector(state, summary)], dtype=torch.float32, device=self.runtime_device),
            )
        predicted_id = int(torch.argmax(logits, dim=-1).item())
        return self.id_to_label.get(predicted_id, "phase_change:NS_GREEN"), state.current_time + max(0.01, math.expm1(float(log_delta.item())))

    def condition_probabilities(self, state: ReplayState, summary: dict[str, Any]) -> dict[str, dict[str, float]]:
        labels = state.context_labels[-self.context_len :]
        deltas = state.context_deltas[-self.context_len :]
        batch = {
            "labels": torch.tensor([[self.label_to_id.get(label, 0) for label in labels]], dtype=torch.long, device=self.runtime_device),
            "deltas": torch.tensor([deltas], dtype=torch.float32, device=self.runtime_device),
            "lengths": torch.tensor([len(labels)], dtype=torch.long, device=self.runtime_device),
            "state_features": torch.tensor([condition_feature_vector(state, summary)], dtype=torch.float32, device=self.runtime_device),
        }
        with torch.no_grad():
            condition_logits = self.condition_head(self._trunk(batch))
        condition_logits = self.apply_platt_scalers(condition_logits, self.condition_scalers)
        probs = torch.sigmoid(condition_logits[0]).tolist()
        output: dict[str, dict[str, float]] = {}
        for horizon_index, horizon in enumerate(self.condition_horizons):
            start = horizon_index * len(CONDITION_NAMES)
            output[f"{int(horizon)}s"] = {
                name: probs[start + name_index]
                for name_index, name in enumerate(CONDITION_NAMES)
            }
        return output

    def predict_time_condition_scores(
        self,
        state: ReplayState,
        summary: dict[str, Any],
        horizons: list[float],
    ) -> dict[str, dict[str, float]] | None:
        requested = {f"{int(horizon)}s" for horizon in horizons}
        return {key: value for key, value in self.condition_probabilities(state, summary).items() if key in requested}

    def predict_time_conditions(
        self,
        state: ReplayState,
        summary: dict[str, Any],
        horizons: list[float],
    ) -> dict[str, dict[str, bool]] | None:
        requested = {f"{int(horizon)}s" for horizon in horizons}
        probs = self.condition_probabilities(state, summary)
        output: dict[str, dict[str, bool]] = {}
        for horizon_index, horizon in enumerate(self.condition_horizons):
            key = f"{int(horizon)}s"
            start = horizon_index * len(CONDITION_NAMES)
            output[key] = {
                name: probs[key][name] >= self.condition_thresholds[start + name_index]
                for name_index, name in enumerate(CONDITION_NAMES)
            }
        return {key: value for key, value in output.items() if key in requested}

    def save_checkpoint(self, path: str | Path) -> None:
        payload = self.checkpoint_base_payload()
        payload["model_state_dict"] = self.model.state_dict()
        payload["condition_head_state_dict"] = self.condition_head.state_dict()
        payload["state_dim"] = int(self.model.shared[0].in_features - self.model.label_head.in_features)
        payload["hidden_dim"] = int(self.model.label_head.in_features)
        payload["embedding_dim"] = int(self.model.embedding.embedding_dim)
        self.save_torch_checkpoint(path, payload)

    @classmethod
    def load_checkpoint(cls, path: str | Path, device: str = "auto") -> "ContinuousTPPBaseline":
        payload = torch.load(path, map_location="cpu")
        model = cls(
            context_len=int(payload["context_len"]),
            epochs=int(payload.get("epochs") or 16),
            batch_size=int(payload.get("batch_size") or 64),
            learning_rate=float(payload.get("learning_rate") or 1e-3),
            device=device if device != "auto" else str(payload.get("runtime_device", "cpu")),
            condition_horizons=tuple(float(value) for value in payload.get("condition_horizons", (10.0, 30.0, 60.0))),
            condition_loss_weight=float(payload.get("condition_loss_weight") or 0.12),
        )
        model.runtime_device = model.resolve_device()
        model.restore_base_state(payload)
        model.model = ContinuousLSTMNextEventModel(
            vocab_size=len(model.label_to_id) + 1,
            state_dim=int(payload["state_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            embedding_dim=int(payload["embedding_dim"]),
        ).to(model.runtime_device)
        model.condition_head = nn.Sequential(
            nn.Linear(int(payload["hidden_dim"]), int(payload["hidden_dim"])),
            nn.ReLU(),
            nn.Linear(int(payload["hidden_dim"]), len(model.condition_horizons) * len(CONDITION_NAMES)),
        ).to(model.runtime_device)
        model.model.load_state_dict(payload["model_state_dict"])
        model.condition_head.load_state_dict(payload["condition_head_state_dict"])
        model.model.eval()
        model.condition_head.eval()
        return model
