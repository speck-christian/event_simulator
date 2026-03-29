from __future__ import annotations

import argparse
import json
from pathlib import Path

from event_simulator.evaluation.prediction_dashboard import build_prediction_dashboard_html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the traffic-and-prediction dashboard from an existing benchmark report and cached runs")
    parser.add_argument("report_json", help="Path to an existing model_comparison.json file")
    parser.add_argument("--cache-dir", required=True, help="Cache directory containing the underlying runs JSON")
    parser.add_argument("--output", default=None, help="Output HTML path. Defaults to traffic_predictions.html next to the report JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = Path(args.report_json)
    report = json.loads(report_path.read_text())
    cache_dir = Path(args.cache_dir)
    seed_start = int(report["example_seed"]) - int(report["train_runs"])
    total_runs = int(report["train_runs"]) + int(report["eval_runs"])
    duration = int(report["duration_seconds"])
    control_mode = report.get("control_mode", "adaptive")
    simulation_profile = report.get("simulation_profile", "richer")
    runs_path = cache_dir / f"runs_{control_mode}_{simulation_profile}_n{total_runs}_d{duration}_seed{seed_start}.json"
    runs = json.loads(runs_path.read_text())
    example_run = next((run for run in runs if int(run["seed"]) == int(report["example_seed"])), None)
    if example_run is None:
        raise SystemExit(f"Could not find example seed {report['example_seed']} in {runs_path}")
    output_path = Path(args.output) if args.output else report_path.with_name("traffic_predictions.html")
    output_path.write_text(build_prediction_dashboard_html(report, example_run, cached_runs=runs))
    print(str(output_path))


if __name__ == "__main__":
    main()
