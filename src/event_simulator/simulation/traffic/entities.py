from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LaneState:
    name: str
    arrival_rate_per_minute: float
    service_headway_seconds: float = 2.0
    turn_share: float = 0.0
    burst_probability: float = 0.0
    burst_gap_scale: float = 1.0
    queue: int = 0
    total_arrivals: int = 0
    total_departures: int = 0
    cumulative_wait_seconds: float = 0.0
    max_queue: int = 0


@dataclass
class EventRecord:
    time_s: float
    event_type: str
    lane: str | None
    detail: str
    queue_after: int | None
    signal_phase: str
