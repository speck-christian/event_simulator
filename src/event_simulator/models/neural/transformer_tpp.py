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
from ..common.networks import AttentionTPPModel
from ..common.replay import ReplayState


class TransformerTPPBaseline(LearnedTPPBaseline):
    name = "transformer_tpp"
    description = "Causal attention temporal point process baseline with position/time encoding and rollout-aware scheduled-sampling training"

    def __init__(
        self,
        context_len: int = 32,
        epochs: int = 18,
        batch_size: int = 64,
        learning_rate: float = 7.5e-4,
        device: str = "auto",
        condition_horizons: tuple[float, ...] = (10.0, 30.0, 60.0),
        condition_loss_weight: float = 0.08,
    ) -> None:
        self.context_len = context_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.rollout_batch_interval = 3
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
                rollout_state = state.clone()
                rollout_state.update(next_event, run["summary"])
                second_event = events[event_index + 2] if event_index + 2 < len(events) else None
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
                        "next_state_features": torch.tensor(condition_feature_vector(rollout_state, run["summary"]), dtype=torch.float32),
                        "second_target_label": self.label_to_id[event_label(second_event)] if second_event is not None else -100,
                        "second_target_delta": (
                            math.log1p(max(0.01, float(second_event["time_s"]) - float(next_event["time_s"])))
                            if second_event is not None
                            else 0.0
                        ),
                        "has_second_target": second_event is not None,
                        "condition_targets": torch.tensor(condition_targets, dtype=torch.float32),
                        "condition_mask": torch.tensor(condition_mask, dtype=torch.float32),
                    }
                )
        return samples

    def fit(self, train_runs: list[dict[str, Any]]) -> None:
        torch.manual_seed(17)
        random.seed(17)
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
        self.model = AttentionTPPModel(
            vocab_size=len(self.label_to_id) + 1,
            state_dim=len(samples[0]["state_features"]),
            hidden_dim=96,
            embedding_dim=32,
            max_len=self.context_len,
            num_heads=4,
            num_layers=2,
            dropout=0.1,
        ).to(self.runtime_device)
        state_dim = len(samples[0]["state_features"])
        self.condition_trunk = nn.Sequential(
            nn.LayerNorm(96 + state_dim),
            nn.Linear(96 + state_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 96),
            nn.GELU(),
        ).to(self.runtime_device)
        self.condition_head = nn.Sequential(
            nn.LayerNorm(96),
            nn.Linear(96, 96),
            nn.ReLU(),
            nn.Linear(96, len(self.condition_horizons) * len(CONDITION_NAMES)),
        ).to(self.runtime_device)
        optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.condition_trunk.parameters()) + list(self.condition_head.parameters()),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )
        label_loss_fn = nn.CrossEntropyLoss()
        delta_loss_fn = nn.SmoothL1Loss()
        pos_counts = torch.stack([sample["condition_targets"] for sample in samples]).sum(dim=0)
        valid_counts = torch.stack([sample["condition_mask"] for sample in samples]).sum(dim=0).clamp(min=1.0)
        neg_counts = (valid_counts - pos_counts).clamp(min=0.0)
        pos_weight = (neg_counts / pos_counts.clamp(min=1.0)).clamp(min=0.5, max=8.0).to(self.runtime_device)
        condition_loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        self.model.train()
        self.condition_trunk.train()
        self.condition_head.train()
        for epoch_index in range(self.epochs):
            rollout_ratio = max(0.0, (epoch_index + 1 - self.epochs * 0.35) / max(1.0, self.epochs * 0.65)) * 0.45
            for batch in loader:
                batch = self.move_batch_to_device(batch, self.runtime_device)
                optimizer.zero_grad()
                trunk = self.model.compute_trunk(batch["labels"], batch["deltas"], batch["lengths"], batch["state_features"])
                logits = self.model.label_head(trunk)
                log_delta = self.model.delta_head(trunk).squeeze(-1)
                condition_logits = self.compute_condition_logits(trunk, batch["state_features"])
                loss = label_loss_fn(logits, batch["target_label"]) + 0.35 * delta_loss_fn(log_delta, batch["target_delta"])
                condition_loss = condition_loss_fn(condition_logits, batch["condition_targets"])
                masked_condition_loss = (condition_loss * batch["condition_mask"]).sum() / batch["condition_mask"].sum().clamp(min=1.0)
                loss = loss + self.condition_loss_weight * masked_condition_loss
                second_mask = batch["has_second_target"]
                use_rollout_loss = second_mask.any() and rollout_ratio > 0.0 and (epoch_index % 2 == 1 or random.random() < (1.0 / self.rollout_batch_interval))
                if use_rollout_loss:
                    with torch.no_grad():
                        predicted_ids = torch.argmax(logits, dim=-1)
                        predicted_deltas = torch.expm1(log_delta).clamp(min=0.01, max=30.0)
                        teacher_deltas = torch.expm1(batch["target_delta"]).clamp(min=0.01, max=30.0)
                        use_model = (torch.rand_like(predicted_deltas) < rollout_ratio) & second_mask
                        rollout_labels, rollout_deltas, rollout_lengths = self.append_rollout_context(
                            batch["labels"],
                            batch["deltas"],
                            batch["lengths"],
                            torch.where(use_model, predicted_ids, batch["target_label"]),
                            torch.where(use_model, predicted_deltas, teacher_deltas),
                        )
                    rollout_logits, rollout_log_delta = self.model(
                        rollout_labels,
                        rollout_deltas,
                        rollout_lengths,
                        batch["next_state_features"],
                    )
                    loss = loss + 0.22 * label_loss_fn(rollout_logits, batch["second_target_label"])
                    loss = loss + 0.08 * delta_loss_fn(rollout_log_delta[second_mask], batch["second_target_delta"][second_mask])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.condition_trunk.parameters()) + list(self.condition_head.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()
        self.model.eval()
        self.condition_trunk.eval()
        self.condition_head.eval()
        self.condition_scalers, self.condition_thresholds = self.tune_condition_calibration(threshold_loader)

    def collate(self, batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        collated = self.collate_sequence_batch(batch, pad_to_context_len=True)
        collated["condition_targets"] = torch.stack([item["condition_targets"] for item in batch])
        collated["condition_mask"] = torch.stack([item["condition_mask"] for item in batch])
        return collated

    def tune_condition_calibration(self, loader: DataLoader) -> tuple[list[tuple[float, float]], list[float]]:
        logits_by_dim = [[] for _ in range(len(self.condition_thresholds))]
        targets_by_dim = [[] for _ in range(len(self.condition_thresholds))]
        with torch.no_grad():
            for batch in loader:
                batch = self.move_batch_to_device(batch, self.runtime_device)
                trunk = self.model.compute_trunk(batch["labels"], batch["deltas"], batch["lengths"], batch["state_features"])
                logits = self.compute_condition_logits(trunk, batch["state_features"]).cpu()
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

    def compute_condition_logits(self, trunk: torch.Tensor, state_features: torch.Tensor) -> torch.Tensor:
        condition_input = torch.cat([trunk, state_features], dim=-1)
        return self.condition_head(self.condition_trunk(condition_input))

    def append_rollout_context(
        self,
        labels: torch.Tensor,
        deltas: torch.Tensor,
        lengths: torch.Tensor,
        appended_labels: torch.Tensor,
        appended_deltas: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rollout_labels = labels.clone()
        rollout_deltas = deltas.clone()
        rollout_lengths = lengths.clone()
        non_full_mask = lengths < self.context_len
        if non_full_mask.any():
            row_indices = torch.arange(labels.size(0))[non_full_mask]
            insert_positions = lengths[non_full_mask]
            rollout_labels[row_indices, insert_positions] = appended_labels[non_full_mask]
            rollout_deltas[row_indices, insert_positions] = appended_deltas[non_full_mask]
            rollout_lengths[non_full_mask] = lengths[non_full_mask] + 1
        full_mask = ~non_full_mask
        if full_mask.any():
            shifted_labels = rollout_labels[full_mask].roll(shifts=-1, dims=1)
            shifted_deltas = rollout_deltas[full_mask].roll(shifts=-1, dims=1)
            shifted_labels[:, -1] = appended_labels[full_mask]
            shifted_deltas[:, -1] = appended_deltas[full_mask]
            rollout_labels[full_mask] = shifted_labels
            rollout_deltas[full_mask] = shifted_deltas
        return rollout_labels, rollout_deltas, rollout_lengths

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
        with torch.no_grad():
            trunk = self.model.compute_trunk(
                torch.tensor([[self.label_to_id.get(label, 0) for label in labels]], dtype=torch.long, device=self.runtime_device),
                torch.tensor([deltas], dtype=torch.float32, device=self.runtime_device),
                torch.tensor([len(labels)], dtype=torch.long, device=self.runtime_device),
                torch.tensor([condition_feature_vector(state, summary)], dtype=torch.float32, device=self.runtime_device),
            )
            state_features = torch.tensor([condition_feature_vector(state, summary)], dtype=torch.float32, device=self.runtime_device)
            condition_logits = self.apply_platt_scalers(self.compute_condition_logits(trunk, state_features), self.condition_scalers)
            probs = torch.sigmoid(condition_logits[0]).tolist()
        output: dict[str, dict[str, float]] = {}
        for horizon_index, horizon in enumerate(self.condition_horizons):
            start = horizon_index * len(CONDITION_NAMES)
            output[f"{int(horizon)}s"] = {name: float(probs[start + idx]) for idx, name in enumerate(CONDITION_NAMES)}
        return output

    def predict_time_condition_scores(self, state: ReplayState, summary: dict[str, Any], horizons: list[float]) -> dict[str, dict[str, float]] | None:
        requested = {f"{int(horizon)}s" for horizon in horizons}
        return {key: value for key, value in self.condition_probabilities(state, summary).items() if key in requested}

    def predict_time_conditions(self, state: ReplayState, summary: dict[str, Any], horizons: list[float]) -> dict[str, dict[str, bool]] | None:
        requested = {f"{int(horizon)}s" for horizon in horizons}
        probs = self.condition_probabilities(state, summary)
        output: dict[str, dict[str, bool]] = {}
        for horizon_index, horizon in enumerate(self.condition_horizons):
            key = f"{int(horizon)}s"
            start = horizon_index * len(CONDITION_NAMES)
            output[key] = {
                name: probs[key][name] >= self.condition_thresholds[start + idx]
                for idx, name in enumerate(CONDITION_NAMES)
            }
        return {key: value for key, value in output.items() if key in requested}

    def save_checkpoint(self, path: str | Path) -> None:
        payload = self.checkpoint_base_payload()
        payload["model_state_dict"] = self.model.state_dict()
        payload["condition_trunk_state_dict"] = self.condition_trunk.state_dict()
        payload["condition_head_state_dict"] = self.condition_head.state_dict()
        payload["state_dim"] = int(self.model.shared[0].in_features - 2 * self.model.hidden_dim)
        payload["hidden_dim"] = int(self.model.hidden_dim)
        payload["embedding_dim"] = int(self.model.embedding.embedding_dim)
        payload["num_heads"] = int(self.model.decoder_blocks[0].attn.num_heads)
        payload["num_layers"] = int(len(self.model.decoder_blocks))
        payload["dropout"] = float(self.model.decoder_blocks[0].attn.dropout)
        self.save_torch_checkpoint(path, payload)

    @classmethod
    def load_checkpoint(cls, path: str | Path, device: str = "auto") -> "TransformerTPPBaseline":
        payload = torch.load(path, map_location="cpu")
        model = cls(
            context_len=int(payload["context_len"]),
            epochs=int(payload.get("epochs") or 18),
            batch_size=int(payload.get("batch_size") or 64),
            learning_rate=float(payload.get("learning_rate") or 7.5e-4),
            device=device if device != "auto" else str(payload.get("runtime_device", "cpu")),
            condition_horizons=tuple(float(value) for value in payload.get("condition_horizons", (10.0, 30.0, 60.0))),
            condition_loss_weight=float(payload.get("condition_loss_weight") or 0.08),
        )
        model.runtime_device = model.resolve_device()
        model.restore_base_state(payload)
        model.model = AttentionTPPModel(
            vocab_size=len(model.label_to_id) + 1,
            state_dim=int(payload["state_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            embedding_dim=int(payload["embedding_dim"]),
            max_len=model.context_len,
            num_heads=int(payload.get("num_heads", 4)),
            num_layers=int(payload.get("num_layers", 2)),
            dropout=float(payload.get("dropout", 0.1)),
        ).to(model.runtime_device)
        state_dim = int(payload["state_dim"])
        model.condition_trunk = nn.Sequential(
            nn.LayerNorm(int(payload["hidden_dim"]) + state_dim),
            nn.Linear(int(payload["hidden_dim"]) + state_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, int(payload["hidden_dim"])),
            nn.GELU(),
        ).to(model.runtime_device)
        model.condition_head = nn.Sequential(
            nn.LayerNorm(int(payload["hidden_dim"])),
            nn.Linear(int(payload["hidden_dim"]), int(payload["hidden_dim"])),
            nn.ReLU(),
            nn.Linear(int(payload["hidden_dim"]), len(model.condition_horizons) * len(CONDITION_NAMES)),
        ).to(model.runtime_device)
        model.model.load_state_dict(payload["model_state_dict"])
        model.condition_trunk.load_state_dict(payload["condition_trunk_state_dict"])
        model.condition_head.load_state_dict(payload["condition_head_state_dict"])
        model.model.eval()
        model.condition_trunk.eval()
        model.condition_head.eval()
        return model
