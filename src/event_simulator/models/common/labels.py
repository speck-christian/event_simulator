from __future__ import annotations

import statistics
from typing import Any


LANES = ("north", "south", "east", "west")
CYCLE_STATES = ("NS_GREEN", "ALL_RED", "EW_GREEN", "ALL_RED")


def parse_phase(detail: str, fallback: str) -> str:
    if detail.startswith("phase="):
        phase_part = detail.split(";", 1)[0]
        return phase_part.split("=", 1)[1]
    return fallback


def event_label(event: dict[str, Any]) -> str:
    if event["event_type"] == "phase_change":
        phase = parse_phase(event["detail"], event["signal_phase"])
        return f"phase_change:{phase}"
    if event["lane"]:
        return f"{event['event_type']}:{event['lane']}"
    return event["event_type"]


def event_family(label: str) -> str:
    return label.split(":", 1)[0]


def mean_or_default(values: list[float], default: float) -> float:
    return statistics.fmean(values) if values else default


def classify_phase_index(previous_index: int | None, phase_name: str) -> int:
    if previous_index is None:
        if phase_name == "NS_GREEN":
            return 0
        if phase_name == "EW_GREEN":
            return 2
        return 1
    return (previous_index + 1) % 4


def next_phase_name(phase_index: int) -> str:
    return CYCLE_STATES[(phase_index + 1) % 4]


def phase_duration(summary: dict[str, Any], phase_index: int) -> float:
    plan = summary["signal_plan"]
    if phase_index == 0:
        return float(plan["NS_GREEN"])
    if phase_index == 2:
        return float(plan["EW_GREEN"])
    return float(plan["ALL_RED"])
