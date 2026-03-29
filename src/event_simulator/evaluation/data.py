from __future__ import annotations

import json
from pathlib import Path

from event_simulator.simulation.traffic import IntersectionSimulation


def infer_control_mode(summary: dict) -> str:
    explicit = summary.get("control_mode")
    if explicit in {"adaptive", "fixed"}:
        return explicit
    green_control = summary.get("signal_plan", {}).get("green_control")
    if isinstance(green_control, dict):
        mode = green_control.get("mode")
        if mode == "queue_and_arrival_rate_responsive":
            return "adaptive"
        if mode == "fixed_time":
            return "fixed"
    return "fixed"


def infer_simulation_profile(summary: dict) -> str:
    profile = summary.get("simulation_profile")
    if profile in {"baseline", "richer"}:
        return profile
    arrival_profile = summary.get("signal_plan", {}).get("arrival_process", {}).get("profile")
    if arrival_profile == "time_varying_bursty_piecewise":
        return "richer"
    return "baseline"


def generate_runs(
    num_runs: int,
    duration: int,
    seed_start: int,
    control_mode: str = "adaptive",
    simulation_profile: str = "richer",
) -> list[dict]:
    runs = []
    for index in range(num_runs):
        sim = IntersectionSimulation(
            duration_seconds=duration,
            seed=seed_start + index,
            control_mode=control_mode,
            simulation_profile=simulation_profile,
        )
        summary = sim.run()
        runs.append(
            {
                "seed": seed_start + index,
                "summary": summary,
                "events": [event.__dict__ for event in sim.records],
            }
        )
    return runs


def load_or_generate_runs(
    num_runs: int,
    duration: int,
    seed_start: int,
    cache_dir: str | None,
    control_mode: str = "adaptive",
    simulation_profile: str = "richer",
) -> list[dict]:
    if not cache_dir:
        return generate_runs(
            num_runs,
            duration,
            seed_start,
            control_mode=control_mode,
            simulation_profile=simulation_profile,
        )
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / f"runs_{control_mode}_{simulation_profile}_n{num_runs}_d{duration}_seed{seed_start}.json"
    if cache_file.exists():
        cached_runs = json.loads(cache_file.read_text())
        if cached_runs:
            cached_mode = infer_control_mode(cached_runs[0].get("summary", {}))
            cached_profile = infer_simulation_profile(cached_runs[0].get("summary", {}))
            if cached_mode == control_mode and cached_profile == simulation_profile:
                return cached_runs
    runs = generate_runs(
        num_runs,
        duration,
        seed_start,
        control_mode=control_mode,
        simulation_profile=simulation_profile,
    )
    cache_file.write_text(json.dumps(runs, indent=2) + "\n")
    return runs
