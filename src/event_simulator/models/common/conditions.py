from __future__ import annotations

from .replay import ReplayState
from .replay import state_feature_vector


CONDITION_NAMES = (
    "congested",
    "severe_queue",
    "ns_pressure_high",
    "ew_pressure_high",
    "pressure_imbalance",
)


def condition_flags(state: ReplayState) -> dict[str, bool]:
    ns_queue = state.queue_state["north"] + state.queue_state["south"]
    ew_queue = state.queue_state["east"] + state.queue_state["west"]
    total_queue = ns_queue + ew_queue
    max_lane_queue = max(state.queue_state.values())
    return {
        "congested": total_queue >= 12,
        "severe_queue": max_lane_queue >= 6,
        "ns_pressure_high": ns_queue >= 8,
        "ew_pressure_high": ew_queue >= 8,
        "pressure_imbalance": abs(ns_queue - ew_queue) >= 5,
    }


def condition_feature_vector(state: ReplayState, summary: dict) -> list[float]:
    base = state_feature_vector(state, summary)
    ns_queue = state.queue_state["north"] + state.queue_state["south"]
    ew_queue = state.queue_state["east"] + state.queue_state["west"]
    total_queue = ns_queue + ew_queue
    max_lane_queue = max(state.queue_state.values())
    imbalance = abs(ns_queue - ew_queue)
    active_ns = 1.0 if state.current_phase == "NS_GREEN" else 0.0
    active_ew = 1.0 if state.current_phase == "EW_GREEN" else 0.0
    return base + [
        min(1.0, ns_queue / 20.0),
        min(1.0, ew_queue / 20.0),
        min(1.0, total_queue / 24.0),
        min(1.0, max_lane_queue / 10.0),
        min(1.0, imbalance / 12.0),
        active_ns,
        active_ew,
    ]
