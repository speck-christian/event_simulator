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

CONGESTED_THRESHOLD = 18
SEVERE_QUEUE_THRESHOLD = 9
NS_PRESSURE_THRESHOLD = 8
EW_PRESSURE_THRESHOLD = 8
PRESSURE_IMBALANCE_THRESHOLD = 5


def condition_flags(state: ReplayState) -> dict[str, bool]:
    ns_queue = state.queue_state["north"] + state.queue_state["south"]
    ew_queue = state.queue_state["east"] + state.queue_state["west"]
    total_queue = ns_queue + ew_queue
    max_lane_queue = max(state.queue_state.values())
    return {
        "congested": total_queue >= CONGESTED_THRESHOLD,
        "severe_queue": max_lane_queue >= SEVERE_QUEUE_THRESHOLD,
        "ns_pressure_high": ns_queue >= NS_PRESSURE_THRESHOLD,
        "ew_pressure_high": ew_queue >= EW_PRESSURE_THRESHOLD,
        "pressure_imbalance": abs(ns_queue - ew_queue) >= PRESSURE_IMBALANCE_THRESHOLD,
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


def symbolic_condition_feature_vector(state: ReplayState) -> list[float]:
    ns_queue = float(state.queue_state["north"] + state.queue_state["south"])
    ew_queue = float(state.queue_state["east"] + state.queue_state["west"])
    total_queue = ns_queue + ew_queue
    max_lane_queue = float(max(state.queue_state.values()))
    imbalance = abs(ns_queue - ew_queue)
    active_ns = 1.0 if state.current_phase == "NS_GREEN" else 0.0
    active_ew = 1.0 if state.current_phase == "EW_GREEN" else 0.0
    ns_share = ns_queue / max(1.0, total_queue)
    ew_share = ew_queue / max(1.0, total_queue)

    # Explicit rule margins mirror the hand-defined condition thresholds.
    congested_margin = (total_queue - float(CONGESTED_THRESHOLD)) / float(CONGESTED_THRESHOLD)
    severe_margin = (max_lane_queue - float(SEVERE_QUEUE_THRESHOLD)) / float(SEVERE_QUEUE_THRESHOLD)
    ns_margin = (ns_queue - float(NS_PRESSURE_THRESHOLD)) / float(NS_PRESSURE_THRESHOLD)
    ew_margin = (ew_queue - float(EW_PRESSURE_THRESHOLD)) / float(EW_PRESSURE_THRESHOLD)
    imbalance_margin = (imbalance - float(PRESSURE_IMBALANCE_THRESHOLD)) / float(PRESSURE_IMBALANCE_THRESHOLD)

    return [
        min(1.0, total_queue / 24.0),
        min(1.0, max_lane_queue / 10.0),
        min(1.0, ns_queue / 20.0),
        min(1.0, ew_queue / 20.0),
        min(1.0, imbalance / 12.0),
        ns_share,
        ew_share,
        active_ns,
        active_ew,
        congested_margin,
        severe_margin,
        ns_margin,
        ew_margin,
        imbalance_margin,
        congested_margin * active_ew,
        severe_margin * active_ew,
        ns_margin * active_ew,
        ew_margin * active_ns,
        imbalance_margin * (active_ns - active_ew),
    ]
