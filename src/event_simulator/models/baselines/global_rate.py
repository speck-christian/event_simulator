from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from ..base import Predictor
from ..common.labels import event_label, mean_or_default
from ..common.replay import ReplayState


class GlobalRateBaseline(Predictor):
    name = "global_rate"
    description = "Global empirical recurrence times per event label"

    def fit(self, train_runs: list[dict[str, Any]]) -> None:
        label_times: dict[str, list[float]] = defaultdict(list)
        all_gaps: list[float] = []
        for run in train_runs:
            for event in run["events"]:
                label_times[event_label(event)].append(float(event["time_s"]))
            times = [float(event["time_s"]) for event in run["events"]]
            all_gaps.extend(max(0.01, b - a) for a, b in zip(times, times[1:]))

        self.mean_gap_by_label = {}
        for label, times in label_times.items():
            gaps = [max(0.01, b - a) for a, b in zip(times, times[1:])]
            self.mean_gap_by_label[label] = mean_or_default(gaps, default=mean_or_default(all_gaps, 5.0))
        self.default_gap = mean_or_default(all_gaps, 5.0)

    def predict(self, state: ReplayState, summary: dict[str, Any]) -> tuple[str, float]:
        best_label = None
        best_time = math.inf
        for label, mean_gap in self.mean_gap_by_label.items():
            last_seen = state.last_seen_by_label.get(label)
            if last_seen is None:
                predicted_time = state.current_time + mean_gap
            else:
                predicted_time = state.current_time + max(0.05, mean_gap - (state.current_time - last_seen))
            if predicted_time < best_time:
                best_time = predicted_time
                best_label = label
        if best_label is None:
            return ("phase_change:NS_GREEN", state.current_time + self.default_gap)
        return best_label, best_time

