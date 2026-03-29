from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..common.datasets import LearnedTPPBaseline, SequenceDataset
from ..common.networks import GRUNextEventModel
from ..common.replay import ReplayState, state_feature_vector


class NeuralTPPBaseline(LearnedTPPBaseline):
    name = "neural_tpp"
    description = "Learned GRU next-event predictor over recent event history plus queue/phase state"

    def __init__(
        self,
        context_len: int = 32,
        epochs: int = 14,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        device: str = "auto",
    ) -> None:
        self.context_len = context_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device

    def fit(self, train_runs: list[dict[str, Any]]) -> None:
        torch.manual_seed(7)
        random.seed(7)
        self.runtime_device = self.resolve_device()
        self.initialize_label_vocab(train_runs)
        samples = self.build_sequence_samples(train_runs)
        loader = DataLoader(SequenceDataset(samples), batch_size=self.batch_size, shuffle=True, collate_fn=self.collate)
        self.model = GRUNextEventModel(
            vocab_size=len(self.label_to_id) + 1,
            state_dim=len(samples[0]["state_features"]),
            hidden_dim=64,
            embedding_dim=24,
        ).to(self.runtime_device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        label_loss_fn = nn.CrossEntropyLoss()
        delta_loss_fn = nn.SmoothL1Loss()
        self.model.train()
        for _ in range(self.epochs):
            for batch in loader:
                batch = self.move_batch_to_device(batch, self.runtime_device)
                optimizer.zero_grad()
                logits, log_delta = self.model(batch["labels"], batch["deltas"], batch["lengths"], batch["state_features"])
                loss = label_loss_fn(logits, batch["target_label"]) + 0.35 * delta_loss_fn(log_delta, batch["target_delta"])
                loss.backward()
                optimizer.step()
        self.model.eval()

    def collate(self, batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        return self.collate_sequence_batch(batch)

    def predict(self, state: ReplayState, summary: dict[str, Any]) -> tuple[str, float]:
        labels = state.context_labels[-self.context_len :]
        deltas = state.context_deltas[-self.context_len :]
        with torch.no_grad():
            logits, log_delta = self.model(
                torch.tensor([[self.label_to_id.get(label, 0) for label in labels]], dtype=torch.long, device=self.runtime_device),
                torch.tensor([deltas], dtype=torch.float32, device=self.runtime_device),
                torch.tensor([len(labels)], dtype=torch.long, device=self.runtime_device),
                torch.tensor([state_feature_vector(state, summary)], dtype=torch.float32, device=self.runtime_device),
            )
        predicted_id = int(torch.argmax(logits, dim=-1).item())
        return self.id_to_label.get(predicted_id, "phase_change:NS_GREEN"), state.current_time + max(0.01, math.expm1(float(log_delta.item())))

    def save_checkpoint(self, path: str | Path) -> None:
        payload = self.checkpoint_base_payload()
        payload["model_state_dict"] = self.model.state_dict()
        payload["state_dim"] = int(self.model.shared[0].in_features - self.model.gru.hidden_size)
        payload["hidden_dim"] = int(self.model.gru.hidden_size)
        payload["embedding_dim"] = int(self.model.embedding.embedding_dim)
        self.save_torch_checkpoint(path, payload)

    @classmethod
    def load_checkpoint(cls, path: str | Path, device: str = "auto") -> "NeuralTPPBaseline":
        payload = torch.load(path, map_location="cpu")
        model = cls(
            context_len=int(payload["context_len"]),
            epochs=int(payload.get("epochs") or 14),
            batch_size=int(payload.get("batch_size") or 64),
            learning_rate=float(payload.get("learning_rate") or 1e-3),
            device=device if device != "auto" else str(payload.get("runtime_device", "cpu")),
        )
        model.runtime_device = model.resolve_device()
        model.restore_base_state(payload)
        model.model = GRUNextEventModel(
            vocab_size=len(model.label_to_id) + 1,
            state_dim=int(payload["state_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            embedding_dim=int(payload["embedding_dim"]),
        ).to(model.runtime_device)
        model.model.load_state_dict(payload["model_state_dict"])
        model.model.eval()
        return model
