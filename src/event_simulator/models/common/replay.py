from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from .labels import LANES, classify_phase_index, event_label, parse_phase


def lane_headway_seconds(summary: dict[str, Any], lane: str) -> float:
    plan = summary["signal_plan"]
    lane_specific = plan.get("lane_service_headway_seconds", {})
    if lane in lane_specific:
        return float(lane_specific[lane])
    return float(plan["service_headway_seconds"])


@dataclass
class ReplayState:
    current_time: float = 0.0
    current_phase: str = "NS_GREEN"
    phase_index: int | None = None
    phase_start_time: float = 0.0
    last_label: str | None = None

    def __post_init__(self) -> None:
        self.queue_state = {lane: 0 for lane in LANES}
        self.last_seen_by_label: dict[str, float] = {}
        self.next_departure_due: dict[str, float | None] = {lane: None for lane in LANES}
        self.context_labels: list[str] = []
        self.context_deltas: list[float] = []

    def lane_has_green(self, lane: str) -> bool:
        if self.current_phase == "NS_GREEN":
            return lane in ("north", "south")
        if self.current_phase == "EW_GREEN":
            return lane in ("east", "west")
        return False

    def update(self, event: dict[str, Any], summary: dict[str, Any]) -> None:
        label = event_label(event)
        time_s = float(event["time_s"])
        delta_s = max(0.0, time_s - self.current_time)
        self.current_time = time_s
        self.last_seen_by_label[label] = time_s
        self.last_label = label
        self.context_labels.append(label)
        self.context_deltas.append(delta_s)

        if event["event_type"] == "phase_change":
            phase_name = parse_phase(event["detail"], event["signal_phase"])
            self.phase_index = classify_phase_index(self.phase_index, phase_name)
            self.current_phase = phase_name
            self.phase_start_time = time_s

            if self.current_phase == "NS_GREEN":
                self.next_departure_due["east"] = None
                self.next_departure_due["west"] = None
                for lane in ("north", "south"):
                    self.next_departure_due[lane] = (
                        time_s + lane_headway_seconds(summary, lane)
                        if self.queue_state[lane] > 0
                        else None
                    )
            elif self.current_phase == "EW_GREEN":
                self.next_departure_due["north"] = None
                self.next_departure_due["south"] = None
                for lane in ("east", "west"):
                    self.next_departure_due[lane] = (
                        time_s + lane_headway_seconds(summary, lane)
                        if self.queue_state[lane] > 0
                        else None
                    )
            else:
                for lane in LANES:
                    self.next_departure_due[lane] = None
            return

        lane = event["lane"]
        if not lane:
            return

        queue_after = int(event["queue_after"])
        self.queue_state[lane] = queue_after
        headway = lane_headway_seconds(summary, lane)

        if event["event_type"] == "vehicle_arrival":
            if self.lane_has_green(lane) and queue_after == 1 and self.next_departure_due[lane] is None:
                self.next_departure_due[lane] = time_s + headway
            return

        if event["event_type"] == "vehicle_departure":
            self.next_departure_due[lane] = None
            if self.lane_has_green(lane) and queue_after > 0:
                self.next_departure_due[lane] = time_s + headway

    def clone(self) -> "ReplayState":
        return copy.deepcopy(self)


def make_synthetic_event(label: str, time_s: float, state: ReplayState) -> dict[str, Any]:
    family, subtype = label.split(":", 1)
    if family == "phase_change":
        return {
            "time_s": time_s,
            "event_type": "phase_change",
            "lane": None,
            "detail": f"phase={subtype}",
            "queue_after": None,
            "signal_phase": subtype,
        }

    lane = subtype
    if family == "vehicle_arrival":
        queue_after = state.queue_state[lane] + 1
    elif family == "vehicle_departure":
        queue_after = max(0, state.queue_state[lane] - 1)
    else:
        queue_after = state.queue_state[lane]

    return {
        "time_s": time_s,
        "event_type": family,
        "lane": lane,
        "detail": "predicted rollout",
        "queue_after": queue_after,
        "signal_phase": state.current_phase,
    }


def rollout_predicted_events(model: Any, state: ReplayState, summary: dict[str, Any], steps: int) -> list[dict[str, Any]]:
    rollout_state = state.clone()
    predictions: list[dict[str, Any]] = []
    for _ in range(steps):
        predicted_label, predicted_time = model.predict(rollout_state, summary)
        predicted_time = max(predicted_time, rollout_state.current_time + 0.01)
        event = make_synthetic_event(predicted_label, predicted_time, rollout_state)
        rollout_state.update(event, summary)
        predictions.append({"label": predicted_label, "time_s": predicted_time})
    return predictions


def state_feature_vector(state: ReplayState, summary: dict[str, Any]) -> list[float]:
    duration = float(summary["duration_seconds"])
    phase_elapsed = max(0.0, state.current_time - state.phase_start_time) / max(1.0, duration)
    phase_flags = [
        1.0 if state.current_phase == "NS_GREEN" else 0.0,
        1.0 if state.current_phase == "EW_GREEN" else 0.0,
        1.0 if state.current_phase == "ALL_RED" else 0.0,
    ]
    queue_features = [min(1.0, state.queue_state[lane] / 20.0) for lane in LANES]
    departure_features = []
    for lane in LANES:
        due = state.next_departure_due[lane]
        if due is None:
            departure_features.append(-1.0)
        else:
            departure_features.append(min(1.0, max(0.0, due - state.current_time) / 10.0))
    return queue_features + phase_flags + [phase_elapsed] + departure_features
