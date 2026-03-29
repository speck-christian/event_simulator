from __future__ import annotations

import math
from typing import Any

from ..base import Predictor
from ..common.labels import LANES, mean_or_default, next_phase_name, phase_duration
from ..common.replay import ReplayState, lane_headway_seconds


class MechanisticBaseline(Predictor):
    name = "mechanistic"
    description = "Phase-aware queue model with empirical arrival rates and controller timing"

    def fit(self, train_runs: list[dict[str, Any]]) -> None:
        self.mean_interarrival = {}
        self.signal_plan = train_runs[0]["summary"]["signal_plan"]
        self.green_control = self.signal_plan.get("green_control", {})
        self.arrival_rate_per_minute = {}
        self.lane_headways = {}
        for lane in LANES:
            interarrivals: list[float] = []
            for run in train_runs:
                arrival_times = [
                    float(event["time_s"])
                    for event in run["events"]
                    if event["event_type"] == "vehicle_arrival" and event["lane"] == lane
                ]
                interarrivals.extend(max(0.01, b - a) for a, b in zip(arrival_times, arrival_times[1:]))
            fallback = 60.0 / max(
                1.0,
                train_runs[0]["summary"]["lanes"][lane]["arrivals"] / (train_runs[0]["summary"]["duration_seconds"] / 60.0),
            )
            self.mean_interarrival[lane] = mean_or_default(interarrivals, fallback)
            self.arrival_rate_per_minute[lane] = 60.0 / max(0.01, self.mean_interarrival[lane])
            self.lane_headways[lane] = lane_headway_seconds(train_runs[0]["summary"], lane)

    def adaptive_phase_duration(self, state: ReplayState, phase_name: str, summary: dict[str, Any]) -> float:
        if phase_name == "ALL_RED":
            return float(summary["signal_plan"]["ALL_RED"])
        control = summary["signal_plan"].get("green_control")
        if not control or control.get("mode") == "fixed_time":
            if phase_name == "NS_GREEN":
                return float(summary["signal_plan"]["NS_GREEN"])
            return float(summary["signal_plan"]["EW_GREEN"])
        if phase_name == "NS_GREEN":
            lanes = ("north", "south")
            base_duration = float(control["base_green_seconds"]["NS_GREEN"])
        else:
            lanes = ("east", "west")
            base_duration = float(control["base_green_seconds"]["EW_GREEN"])
        total_queue = sum(state.queue_state[lane] for lane in lanes)
        total_arrival_rate = sum(self.arrival_rate_per_minute[lane] for lane in lanes)
        duration = (
            base_duration
            + float(control["queue_weight_seconds"]) * (total_queue - 4)
            + float(control["arrival_weight_seconds"]) * (total_arrival_rate - 20)
        )
        return max(float(control["min_green_seconds"]), min(float(control["max_green_seconds"]), duration))

    def predict(self, state: ReplayState, summary: dict[str, Any]) -> tuple[str, float]:
        candidates: list[tuple[float, str]] = []
        if state.phase_index is not None:
            phase_end = state.phase_start_time + self.adaptive_phase_duration(state, state.current_phase, summary)
            candidates.append((phase_end, f"phase_change:{next_phase_name(state.phase_index)}"))
        for lane in LANES:
            last_seen = state.last_seen_by_label.get(f"vehicle_arrival:{lane}")
            next_arrival = state.current_time + self.mean_interarrival[lane] if last_seen is None else max(
                state.current_time + 0.01,
                last_seen + self.mean_interarrival[lane],
            )
            candidates.append((next_arrival, f"vehicle_arrival:{lane}"))
        for lane, due in state.next_departure_due.items():
            if due is not None and due >= state.current_time:
                candidates.append((due, f"vehicle_departure:{lane}"))
        best_time, best_label = min(candidates, key=lambda item: (item[0], item[1]))
        return best_label, best_time

    def predict_time_condition_scores(
        self,
        state: ReplayState,
        summary: dict[str, Any],
        horizons: list[float],
    ) -> dict[str, dict[str, float]] | None:
        scores: dict[str, dict[str, float]] = {}
        ns_queue = state.queue_state["north"] + state.queue_state["south"]
        ew_queue = state.queue_state["east"] + state.queue_state["west"]
        total_queue = ns_queue + ew_queue
        max_lane_queue = max(state.queue_state.values())
        net_ns_growth = (self.arrival_rate_per_minute["north"] + self.arrival_rate_per_minute["south"]) / 60.0
        net_ew_growth = (self.arrival_rate_per_minute["east"] + self.arrival_rate_per_minute["west"]) / 60.0
        ns_service = 1.0 / max(1.0, self.lane_headways["north"] + self.lane_headways["south"])
        ew_service = 1.0 / max(1.0, self.lane_headways["east"] + self.lane_headways["west"])
        for horizon in horizons:
            scale = max(0.5, horizon / 30.0)
            projected_ns = ns_queue + horizon * (net_ns_growth - (ns_service if state.current_phase == "NS_GREEN" else 0.03))
            projected_ew = ew_queue + horizon * (net_ew_growth - (ew_service if state.current_phase == "EW_GREEN" else 0.03))
            projected_total = max(0.0, projected_ns + projected_ew)
            projected_max_lane = max_lane_queue + horizon * max(0.0, max(net_ns_growth, net_ew_growth) - 0.07)
            imbalance = abs(projected_ns - projected_ew)
            scores[f"{int(horizon)}s"] = {
                "congested": 1.0 / (1.0 + math.exp(-(projected_total - 12.0) / (1.8 * scale))),
                "severe_queue": 1.0 / (1.0 + math.exp(-(projected_max_lane - 6.0) / (1.1 * scale))),
                "ns_pressure_high": 1.0 / (1.0 + math.exp(-(projected_ns - 8.0) / (1.5 * scale))),
                "ew_pressure_high": 1.0 / (1.0 + math.exp(-(projected_ew - 8.0) / (1.5 * scale))),
                "pressure_imbalance": 1.0 / (1.0 + math.exp(-(imbalance - 5.0) / (1.2 * scale))),
            }
        return scores

    def predict_time_conditions(
        self,
        state: ReplayState,
        summary: dict[str, Any],
        horizons: list[float],
    ) -> dict[str, dict[str, bool]] | None:
        scores = self.predict_time_condition_scores(state, summary, horizons)
        if scores is None:
            return None
        return {
            horizon: {name: value >= 0.5 for name, value in score_map.items()}
            for horizon, score_map in scores.items()
        }
