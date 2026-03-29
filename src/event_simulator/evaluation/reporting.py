from __future__ import annotations

from typing import Any

from event_simulator.models.common import event_label


def build_report(train_runs: list[dict[str, Any]], eval_runs: list[dict[str, Any]], model_outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    first_eval = eval_runs[0]
    actual_events = [
        {
            "time_s": float(event["time_s"]),
            "label": event_label(event),
        }
        for event in first_eval["events"][1:]
    ]
    return {
        "train_runs": len(train_runs),
        "eval_runs": len(eval_runs),
        "duration_seconds": first_eval["summary"]["duration_seconds"],
        "control_mode": first_eval["summary"].get("control_mode", "fixed"),
        "simulation_profile": first_eval["summary"].get("simulation_profile", "baseline"),
        "example_seed": first_eval["seed"],
        "actual_events": actual_events,
        "models": model_outputs,
    }
