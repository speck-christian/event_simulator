from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from ..base import Predictor
from ..common.labels import event_label, mean_or_default
from ..common.replay import ReplayState


class TransitionBaseline(Predictor):
    name = "transition"
    description = "Most likely next label and delay conditioned on the current label"

    def fit(self, train_runs: list[dict[str, Any]]) -> None:
        self.next_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.next_deltas: dict[tuple[str, str], list[float]] = defaultdict(list)
        self.global_counts: Counter[str] = Counter()
        self.global_deltas: dict[str, list[float]] = defaultdict(list)
        for run in train_runs:
            for current, nxt in zip(run["events"], run["events"][1:]):
                current_label = event_label(current)
                next_label = event_label(nxt)
                delta = max(0.01, float(nxt["time_s"]) - float(current["time_s"]))
                self.next_counts[current_label][next_label] += 1
                self.next_deltas[(current_label, next_label)].append(delta)
                self.global_counts[next_label] += 1
                self.global_deltas[next_label].append(delta)

    def predict(self, state: ReplayState, summary: dict[str, Any]) -> tuple[str, float]:
        if state.last_label and self.next_counts[state.last_label]:
            predicted_label, _ = self.next_counts[state.last_label].most_common(1)[0]
            return predicted_label, state.current_time + mean_or_default(self.next_deltas[(state.last_label, predicted_label)], 1.0)
        predicted_label, _ = self.global_counts.most_common(1)[0]
        return predicted_label, state.current_time + mean_or_default(self.global_deltas[predicted_label], 1.0)

