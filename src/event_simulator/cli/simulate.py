from __future__ import annotations

import argparse
import json
from pathlib import Path

from event_simulator.simulation.traffic import IntersectionSimulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Traffic intersection control simulation")
    parser.add_argument("--duration", type=int, default=300, help="Simulation duration in seconds")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducible arrivals")
    parser.add_argument("--output-dir", default="output", help="Directory where summary.json and events.csv will be written")
    parser.add_argument(
        "--control-mode",
        choices=("adaptive", "fixed"),
        default="adaptive",
        help="Signal control mode. Adaptive is the default; fixed restores the original static phase timing.",
    )
    parser.add_argument(
        "--simulation-profile",
        choices=("baseline", "richer"),
        default="richer",
        help="Traffic demand/service profile. 'richer' adds burstier arrivals and lane-specific service rates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim = IntersectionSimulation(
        duration_seconds=args.duration,
        seed=args.seed,
        control_mode=args.control_mode,
        simulation_profile=args.simulation_profile,
    )
    summary = sim.run()
    sim.write_outputs(Path(args.output_dir), summary)
    print(json.dumps(summary, indent=2))
