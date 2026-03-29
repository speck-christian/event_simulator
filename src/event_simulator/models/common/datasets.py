from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from ..base import Predictor
from .replay import ReplayState, state_feature_vector
from .labels import event_label


class SequenceDataset(Dataset):
    def __init__(self, samples: list[dict[str, object]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.samples[index]


class LearnedTPPBaseline(Predictor):
    context_len: int
    label_to_id: dict[str, int]
    id_to_label: dict[int, str]
    device: str

    def initialize_label_vocab(self, train_runs: list[dict[str, object]]) -> None:
        labels = sorted({event_label(event) for run in train_runs for event in run["events"]})
        self.label_to_id = {label: index + 1 for index, label in enumerate(labels)}
        self.id_to_label = {index: label for label, index in self.label_to_id.items()}

    def resolve_device(self) -> torch.device:
        requested = getattr(self, "device", "auto")
        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(requested)

    def move_batch_to_device(self, batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for key, value in batch.items():
            moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
        return moved

    def split_train_validation_samples(
        self,
        samples: list[dict[str, object]],
        validation_fraction: float = 0.15,
        seed: int = 7,
        group_key: str | None = None,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        if len(samples) < 8 or validation_fraction <= 0.0:
            return samples, samples
        rng = random.Random(seed)
        if group_key is None:
            indices = list(range(len(samples)))
            rng.shuffle(indices)
            validation_count = max(1, min(len(samples) // 4, int(round(len(samples) * validation_fraction))))
            validation_indices = set(indices[:validation_count])
            train_subset = [sample for index, sample in enumerate(samples) if index not in validation_indices]
            validation_subset = [sample for index, sample in enumerate(samples) if index in validation_indices]
        else:
            groups = sorted({sample[group_key] for sample in samples})
            if len(groups) < 2:
                return samples, samples
            rng.shuffle(groups)
            validation_group_count = max(1, min(len(groups) // 3, int(round(len(groups) * validation_fraction))))
            validation_groups = set(groups[:validation_group_count])
            train_subset = [sample for sample in samples if sample[group_key] not in validation_groups]
            validation_subset = [sample for sample in samples if sample[group_key] in validation_groups]
        if not train_subset or not validation_subset:
            return samples, samples
        return train_subset, validation_subset

    def build_sequence_samples(self, train_runs: list[dict[str, object]], include_rollout_targets: bool = False) -> list[dict[str, object]]:
        samples: list[dict[str, object]] = []
        for run in train_runs:
            state = ReplayState()
            events = run["events"]
            for event_index, (current_event, next_event) in enumerate(zip(events, events[1:])):
                state.update(current_event, run["summary"])
                history_labels = state.context_labels[-self.context_len :]
                history_deltas = state.context_deltas[-self.context_len :]
                target_delta = max(0.01, float(next_event["time_s"]) - state.current_time)
                sample: dict[str, object] = {
                    "labels": torch.tensor([self.label_to_id[label] for label in history_labels], dtype=torch.long),
                    "deltas": torch.tensor(history_deltas, dtype=torch.float32),
                    "state_features": torch.tensor(state_feature_vector(state, run["summary"]), dtype=torch.float32),
                    "target_label": self.label_to_id[event_label(next_event)],
                    "target_delta": math.log1p(target_delta),
                }
                if include_rollout_targets:
                    rollout_state = state.clone()
                    rollout_state.update(next_event, run["summary"])
                    second_event = events[event_index + 2] if event_index + 2 < len(events) else None
                    sample.update(
                        {
                            "next_state_features": torch.tensor(
                                state_feature_vector(rollout_state, run["summary"]),
                                dtype=torch.float32,
                            ),
                            "second_target_label": self.label_to_id[event_label(second_event)] if second_event is not None else -100,
                            "second_target_delta": (
                                math.log1p(max(0.01, float(second_event["time_s"]) - float(next_event["time_s"])))
                                if second_event is not None
                                else 0.0
                            ),
                            "has_second_target": second_event is not None,
                        }
                    )
                samples.append(sample)
        return samples

    def collate_sequence_batch(self, batch: list[dict[str, object]], pad_to_context_len: bool = False) -> dict[str, torch.Tensor]:
        labels = pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=0)
        deltas = pad_sequence([item["deltas"] for item in batch], batch_first=True, padding_value=0.0)
        if pad_to_context_len and labels.size(1) < self.context_len:
            pad_width = self.context_len - labels.size(1)
            labels = torch.cat([labels, torch.zeros(labels.size(0), pad_width, dtype=labels.dtype)], dim=1)
            deltas = torch.cat([deltas, torch.zeros(deltas.size(0), pad_width, dtype=deltas.dtype)], dim=1)

        collated = {
            "labels": labels,
            "deltas": deltas,
            "lengths": torch.tensor([len(item["labels"]) for item in batch], dtype=torch.long),
            "state_features": torch.stack([item["state_features"] for item in batch]),
            "target_label": torch.tensor([item["target_label"] for item in batch], dtype=torch.long),
            "target_delta": torch.tensor([item["target_delta"] for item in batch], dtype=torch.float32),
        }
        if "next_state_features" in batch[0]:
            collated.update(
                {
                    "next_state_features": torch.stack([item["next_state_features"] for item in batch]),
                    "second_target_label": torch.tensor([item["second_target_label"] for item in batch], dtype=torch.long),
                    "second_target_delta": torch.tensor([item["second_target_delta"] for item in batch], dtype=torch.float32),
                    "has_second_target": torch.tensor([item["has_second_target"] for item in batch], dtype=torch.bool),
                }
            )
        return collated

    def fit_platt_scalers(
        self,
        logits_by_dim: list[list[float]],
        targets_by_dim: list[list[float]],
        steps: int = 120,
    ) -> list[tuple[float, float]]:
        scalers: list[tuple[float, float]] = []
        for logits, targets in zip(logits_by_dim, targets_by_dim):
            if not logits:
                scalers.append((1.0, 0.0))
                continue
            target_values = set(float(target) for target in targets)
            if len(target_values) < 2:
                scalers.append((1.0, 0.0))
                continue
            logits_tensor = torch.tensor(logits, dtype=torch.float32)
            targets_tensor = torch.tensor(targets, dtype=torch.float32)
            scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            bias = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            optimizer = torch.optim.Adam([scale, bias], lr=0.05)
            for _ in range(steps):
                optimizer.zero_grad()
                calibrated_logits = logits_tensor * scale + bias
                loss = F.binary_cross_entropy_with_logits(calibrated_logits, targets_tensor)
                loss.backward()
                optimizer.step()
            scalers.append((float(scale.detach().item()), float(bias.detach().item())))
        return scalers

    def apply_platt_scalers(self, logits: torch.Tensor, scalers: list[tuple[float, float]]) -> torch.Tensor:
        if not scalers:
            return logits
        scale = torch.tensor([item[0] for item in scalers], dtype=logits.dtype, device=logits.device)
        bias = torch.tensor([item[1] for item in scalers], dtype=logits.dtype, device=logits.device)
        return logits * scale + bias

    def tune_condition_thresholds_from_probs(
        self,
        probs_by_dim: list[list[float]],
        targets_by_dim: list[list[float]],
        candidate_thresholds: list[float] | None = None,
    ) -> list[float]:
        tuned: list[float] = []
        for probs, targets in zip(probs_by_dim, targets_by_dim):
            if not probs:
                tuned.append(0.5)
                continue
            thresholds = candidate_thresholds or self.build_condition_threshold_candidates(probs)
            best_threshold = 0.5
            best_score = -1.0
            best_f1 = -1.0
            for threshold in thresholds:
                tp = tn = fp = fn = 0
                for prob, target in zip(probs, targets):
                    predicted = prob >= threshold
                    actual = bool(target)
                    if predicted and actual:
                        tp += 1
                    elif predicted and not actual:
                        fp += 1
                    elif not predicted and actual:
                        fn += 1
                    else:
                        tn += 1
                recall = tp / max(1, tp + fn)
                specificity = tn / max(1, tn + fp)
                balanced_accuracy = 0.5 * (recall + specificity)
                precision = tp / max(1, tp + fp)
                f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
                if balanced_accuracy > best_score + 1e-9:
                    best_score = balanced_accuracy
                    best_f1 = f1
                    best_threshold = threshold
                elif abs(balanced_accuracy - best_score) <= 1e-9:
                    if f1 > best_f1 + 1e-9:
                        best_f1 = f1
                        best_threshold = threshold
                    elif abs(f1 - best_f1) <= 1e-9 and abs(threshold - 0.5) < abs(best_threshold - 0.5):
                        best_threshold = threshold
            tuned.append(best_threshold)
        return tuned

    def build_condition_threshold_candidates(self, probs: list[float], max_unique_points: int = 64) -> list[float]:
        base_grid = [round(step * 0.05, 2) for step in range(2, 19)]
        if not probs:
            return base_grid
        unique = sorted({min(0.99, max(0.01, round(float(prob), 4))) for prob in probs})
        if len(unique) > max_unique_points:
            sampled = []
            for index in range(max_unique_points):
                source_index = round(index * (len(unique) - 1) / max(1, max_unique_points - 1))
                sampled.append(unique[source_index])
            unique = sorted(set(sampled))
        midpoints = [
            round((left + right) * 0.5, 4)
            for left, right in zip(unique, unique[1:])
            if right - left >= 1e-4
        ]
        candidates = sorted(
            {
                *base_grid,
                0.5,
                *unique,
                *midpoints,
            }
        )
        return [threshold for threshold in candidates if 0.01 <= threshold <= 0.99]

    def checkpoint_base_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "class_name": self.__class__.__name__,
            "name": getattr(self, "name", self.__class__.__name__),
            "context_len": self.context_len,
            "epochs": getattr(self, "epochs", None),
            "batch_size": getattr(self, "batch_size", None),
            "learning_rate": getattr(self, "learning_rate", None),
            "device": getattr(self, "device", "auto"),
            "runtime_device": str(getattr(self, "runtime_device", "cpu")),
            "label_to_id": getattr(self, "label_to_id", {}),
            "id_to_label": getattr(self, "id_to_label", {}),
        }
        if hasattr(self, "condition_horizons"):
            payload["condition_horizons"] = list(getattr(self, "condition_horizons"))
        if hasattr(self, "condition_loss_weight"):
            payload["condition_loss_weight"] = getattr(self, "condition_loss_weight")
        if hasattr(self, "condition_thresholds"):
            payload["condition_thresholds"] = list(getattr(self, "condition_thresholds"))
        if hasattr(self, "condition_scalers"):
            payload["condition_scalers"] = [list(item) for item in getattr(self, "condition_scalers")]
        if hasattr(self, "validation_fraction"):
            payload["validation_fraction"] = getattr(self, "validation_fraction")
        return payload

    def restore_base_state(self, payload: dict[str, Any]) -> None:
        self.label_to_id = {str(key): int(value) for key, value in payload["label_to_id"].items()}
        self.id_to_label = {int(key): str(value) for key, value in payload["id_to_label"].items()}
        self.runtime_device = torch.device(payload.get("runtime_device", "cpu"))
        if "condition_thresholds" in payload:
            self.condition_thresholds = [float(value) for value in payload["condition_thresholds"]]
        if "condition_scalers" in payload:
            self.condition_scalers = [(float(scale), float(bias)) for scale, bias in payload["condition_scalers"]]

    def save_torch_checkpoint(self, path: str | Path, payload: dict[str, Any]) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, checkpoint_path)
