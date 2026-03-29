from __future__ import annotations

import heapq
import math
import random
from collections import deque
from pathlib import Path
from typing import Callable

from .entities import EventRecord, LaneState
from .io import write_outputs


class IntersectionSimulation:
    def __init__(
        self,
        duration_seconds: int,
        seed: int,
        ns_green_seconds: int = 30,
        all_red_seconds: int = 4,
        ew_green_seconds: int = 25,
        saturation_flow_seconds: int = 2,
        control_mode: str = "adaptive",
        simulation_profile: str = "richer",
    ) -> None:
        self.duration_seconds = duration_seconds
        self.random = random.Random(seed)
        self.ns_green_seconds = ns_green_seconds
        self.all_red_seconds = all_red_seconds
        self.ew_green_seconds = ew_green_seconds
        self.saturation_flow_seconds = saturation_flow_seconds
        if control_mode not in {"adaptive", "fixed"}:
            raise ValueError(f"Unsupported control_mode: {control_mode}")
        if simulation_profile not in {"baseline", "richer"}:
            raise ValueError(f"Unsupported simulation_profile: {simulation_profile}")
        self.control_mode = control_mode
        self.simulation_profile = simulation_profile
        self.min_green_seconds = 12
        self.max_green_seconds = 52
        self.queue_weight_seconds = 1.45
        self.arrival_weight_seconds = 0.28
        self.lanes = self._build_lane_states()
        self.ns_lanes = ("north", "south")
        self.ew_lanes = ("east", "west")
        self.phase_plan = [
            ("NS_GREEN", self.ns_green_seconds),
            ("ALL_RED", self.all_red_seconds),
            ("EW_GREEN", self.ew_green_seconds),
            ("ALL_RED", self.all_red_seconds),
        ]
        self.current_phase_index = 0
        self.current_phase = self.phase_plan[0][0]
        self.events: list[tuple[float, int, Callable[[], None]]] = []
        self.event_counter = 0
        self.records: list[EventRecord] = []
        self.waiting_since = {lane: deque() for lane in self.lanes}
        self.pending_departure = {lane: False for lane in self.lanes}
        self.current_time_s = 0.0
        self.phase_duration_history = {
            "NS_GREEN": [],
            "EW_GREEN": [],
            "ALL_RED": [],
        }

    def _build_lane_states(self) -> dict[str, LaneState]:
        if self.simulation_profile == "baseline":
            return {
                "north": LaneState("north", arrival_rate_per_minute=16, service_headway_seconds=float(self.saturation_flow_seconds)),
                "south": LaneState("south", arrival_rate_per_minute=14, service_headway_seconds=float(self.saturation_flow_seconds)),
                "east": LaneState("east", arrival_rate_per_minute=10, service_headway_seconds=float(self.saturation_flow_seconds)),
                "west": LaneState("west", arrival_rate_per_minute=8, service_headway_seconds=float(self.saturation_flow_seconds)),
            }
        return {
            "north": LaneState(
                "north",
                arrival_rate_per_minute=14,
                service_headway_seconds=1.95,
                turn_share=0.16,
                burst_probability=0.12,
                burst_gap_scale=0.58,
            ),
            "south": LaneState(
                "south",
                arrival_rate_per_minute=12,
                service_headway_seconds=2.1,
                turn_share=0.2,
                burst_probability=0.1,
                burst_gap_scale=0.62,
            ),
            "east": LaneState(
                "east",
                arrival_rate_per_minute=9,
                service_headway_seconds=2.35,
                turn_share=0.28,
                burst_probability=0.09,
                burst_gap_scale=0.68,
            ),
            "west": LaneState(
                "west",
                arrival_rate_per_minute=7,
                service_headway_seconds=2.5,
                turn_share=0.32,
                burst_probability=0.08,
                burst_gap_scale=0.72,
            ),
        }

    def schedule(self, time_s: float, callback: Callable[[], None]) -> None:
        if time_s <= self.duration_seconds:
            heapq.heappush(self.events, (time_s, self.event_counter, callback))
            self.event_counter += 1

    def arrival_rate_at(self, lane: LaneState, time_s: float) -> float:
        if self.simulation_profile == "baseline":
            return lane.arrival_rate_per_minute
        progress = time_s / max(1.0, float(self.duration_seconds))
        wave = 1.0 + 0.16 * math.sin(2.0 * math.pi * progress)
        if lane.name in self.ns_lanes:
            corridor_bias = 1.1 if 0.2 <= progress <= 0.42 else 0.84
        else:
            corridor_bias = 1.08 if 0.56 <= progress <= 0.78 else 0.86
        return max(1.0, lane.arrival_rate_per_minute * wave * corridor_bias)

    def sample_interarrival(self, lane: LaneState, time_s: float) -> float:
        base_gap = self.random.expovariate(self.arrival_rate_at(lane, time_s) / 60.0)
        if self.simulation_profile == "richer" and self.random.random() < lane.burst_probability:
            return max(0.2, base_gap * lane.burst_gap_scale)
        return base_gap

    def service_headway_for(self, lane_name: str) -> float:
        lane = self.lanes[lane_name]
        if self.simulation_profile == "baseline":
            return float(self.saturation_flow_seconds)
        queue_pressure = min(4.0, lane.queue / 8.0)
        phase_bonus = -0.08 if (
            (self.current_phase == "NS_GREEN" and lane_name in self.ns_lanes)
            or (self.current_phase == "EW_GREEN" and lane_name in self.ew_lanes)
        ) else 0.0
        if self.lane_has_green(lane_name):
            phase_bonus = -0.18
        turn_penalty = 0.7 * lane.turn_share
        return max(1.35, lane.service_headway_seconds + 0.12 * queue_pressure + turn_penalty + phase_bonus)

    def log(self, time_s: float, event_type: str, lane: str | None, detail: str) -> None:
        queue_after = self.lanes[lane].queue if lane else None
        self.records.append(
            EventRecord(
                time_s=round(time_s, 3),
                event_type=event_type,
                lane=lane,
                detail=detail,
                queue_after=queue_after,
                signal_phase=self.current_phase,
            )
        )

    def start(self) -> None:
        self.schedule(0.0, lambda: self.change_phase(0.0))
        for lane_name, lane in self.lanes.items():
            self.schedule(self.sample_interarrival(lane, 0.0), lambda ln=lane_name: self.handle_arrival(ln))

    def corridor_pressure(self, corridor: str, time_s: float | None = None) -> tuple[int, float]:
        lane_names = self.ns_lanes if corridor == "NS_GREEN" else self.ew_lanes
        total_queue = sum(self.lanes[lane_name].queue for lane_name in lane_names)
        if time_s is None:
            time_s = self.current_time_s
        total_arrival_rate = sum(self.arrival_rate_at(self.lanes[lane_name], time_s) for lane_name in lane_names)
        return total_queue, total_arrival_rate

    def phase_duration_for(self, phase_name: str) -> float:
        if phase_name == "ALL_RED":
            return float(self.all_red_seconds)
        if self.control_mode == "fixed":
            if phase_name == "NS_GREEN":
                return float(self.ns_green_seconds)
            return float(self.ew_green_seconds)
        if phase_name == "NS_GREEN":
            base_duration = float(self.ns_green_seconds)
        else:
            base_duration = float(self.ew_green_seconds)
        total_queue, total_arrival_rate = self.corridor_pressure(phase_name, self.current_time_s)
        duration = (
            base_duration
            + self.queue_weight_seconds * (total_queue - 4)
            + self.arrival_weight_seconds * (total_arrival_rate - 20)
        )
        return max(float(self.min_green_seconds), min(float(self.max_green_seconds), duration))

    def change_phase(self, time_s: float) -> None:
        self.current_phase = self.phase_plan[self.current_phase_index][0]
        duration = self.phase_duration_for(self.current_phase)
        self.phase_duration_history[self.current_phase].append(duration)
        if self.current_phase == "ALL_RED":
            detail = f"phase={self.current_phase};duration_s={duration:.2f};control_mode={self.control_mode}"
        else:
            if self.control_mode == "adaptive":
                total_queue, total_arrival_rate = self.corridor_pressure(self.current_phase, time_s)
                detail = (
                    f"phase={self.current_phase};duration_s={duration:.2f};control_mode={self.control_mode};"
                    f"queue_pressure={total_queue};arrival_rate_per_min={total_arrival_rate:.2f}"
                )
            else:
                detail = f"phase={self.current_phase};duration_s={duration:.2f};control_mode={self.control_mode}"
        self.log(time_s, "phase_change", None, detail)
        if self.current_phase == "NS_GREEN":
            for lane in self.ns_lanes:
                self.schedule_departure_if_needed(time_s, lane)
        elif self.current_phase == "EW_GREEN":
            for lane in self.ew_lanes:
                self.schedule_departure_if_needed(time_s, lane)
        next_index = (self.current_phase_index + 1) % len(self.phase_plan)
        next_time = time_s + duration
        self.schedule(next_time, lambda nt=next_time, ni=next_index: self._advance_phase(nt, ni))

    def _advance_phase(self, time_s: float, next_index: int) -> None:
        self.current_phase_index = next_index
        self.change_phase(time_s)

    def lane_has_green(self, lane_name: str) -> bool:
        if self.current_phase == "NS_GREEN":
            return lane_name in self.ns_lanes
        if self.current_phase == "EW_GREEN":
            return lane_name in self.ew_lanes
        return False

    def handle_arrival(self, lane_name: str) -> None:
        current_time = self.current_time_s
        lane = self.lanes[lane_name]
        lane.queue += 1
        lane.total_arrivals += 1
        lane.max_queue = max(lane.max_queue, lane.queue)
        self.waiting_since[lane_name].append(current_time)
        self.log(current_time, "vehicle_arrival", lane_name, "vehicle joined queue")
        self.schedule(current_time + self.sample_interarrival(lane, current_time), lambda ln=lane_name: self.handle_arrival(ln))
        if self.lane_has_green(lane_name) and lane.queue == 1:
            self.schedule_departure_if_needed(current_time, lane_name)

    def schedule_departure_if_needed(self, time_s: float, lane_name: str) -> None:
        lane = self.lanes[lane_name]
        if lane.queue <= 0 or not self.lane_has_green(lane_name) or self.pending_departure[lane_name]:
            return
        self.pending_departure[lane_name] = True
        departure_time = time_s + self.service_headway_for(lane_name)
        self.schedule(departure_time, lambda ln=lane_name, dt=departure_time: self.handle_departure(ln, dt))

    def handle_departure(self, lane_name: str, time_s: float) -> None:
        self.pending_departure[lane_name] = False
        lane = self.lanes[lane_name]
        if lane.queue <= 0 or not self.lane_has_green(lane_name):
            return
        lane.queue -= 1
        lane.total_departures += 1
        arrival_time = self.waiting_since[lane_name].popleft()
        lane.cumulative_wait_seconds += max(0.0, time_s - arrival_time)
        self.log(time_s, "vehicle_departure", lane_name, "vehicle cleared intersection")
        if lane.queue > 0:
            self.schedule_departure_if_needed(time_s, lane_name)

    def run(self) -> dict:
        self.start()
        while self.events:
            time_s, _, callback = heapq.heappop(self.events)
            if time_s > self.duration_seconds:
                break
            self.current_time_s = time_s
            callback()
        return self.summary()

    def summary(self) -> dict:
        lane_summaries = {}
        for lane_name, lane in self.lanes.items():
            avg_wait = lane.cumulative_wait_seconds / lane.total_departures if lane.total_departures else 0.0
            lane_summaries[lane_name] = {
                "arrival_rate_per_minute": lane.arrival_rate_per_minute,
                "service_headway_seconds": round(lane.service_headway_seconds, 2),
                "turn_share": round(lane.turn_share, 2),
                "burst_probability": round(lane.burst_probability, 2),
                "arrivals": lane.total_arrivals,
                "departures": lane.total_departures,
                "queue_remaining": lane.queue,
                "max_queue": lane.max_queue,
                "average_wait_seconds": round(avg_wait, 2),
            }
        signal_plan = {
            "NS_GREEN": round(
                sum(self.phase_duration_history["NS_GREEN"]) / max(1, len(self.phase_duration_history["NS_GREEN"])), 2
            ),
            "ALL_RED": round(
                sum(self.phase_duration_history["ALL_RED"]) / max(1, len(self.phase_duration_history["ALL_RED"])), 2
            ),
            "EW_GREEN": round(
                sum(self.phase_duration_history["EW_GREEN"]) / max(1, len(self.phase_duration_history["EW_GREEN"])), 2
            ),
            "service_headway_seconds": round(
                sum(lane.service_headway_seconds for lane in self.lanes.values()) / max(1, len(self.lanes)),
                2,
            ),
            "lane_service_headway_seconds": {
                lane_name: round(lane.service_headway_seconds, 2) for lane_name, lane in self.lanes.items()
            },
            "green_control": {
                "mode": "queue_and_arrival_rate_responsive" if self.control_mode == "adaptive" else "fixed_time",
                "min_green_seconds": self.min_green_seconds,
                "max_green_seconds": self.max_green_seconds,
                "queue_weight_seconds": self.queue_weight_seconds,
                "arrival_weight_seconds": self.arrival_weight_seconds,
                "base_green_seconds": {
                    "NS_GREEN": self.ns_green_seconds,
                    "EW_GREEN": self.ew_green_seconds,
                },
            },
            "arrival_process": {
                "profile": "time_varying_bursty_piecewise" if self.simulation_profile == "richer" else "stationary_poisson",
            },
        }
        return {
            "duration_seconds": self.duration_seconds,
            "control_mode": self.control_mode,
            "simulation_profile": self.simulation_profile,
            "signal_plan": signal_plan,
            "lanes": lane_summaries,
            "events_recorded": len(self.records),
        }

    def write_outputs(self, output_dir: Path, summary: dict) -> None:
        write_outputs(output_dir, summary, self.records)
