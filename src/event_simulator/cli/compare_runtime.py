from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two runtime_summary.json files and report speedups")
    parser.add_argument("baseline", help="Path to the baseline runtime_summary.json, e.g. local CPU run")
    parser.add_argument("candidate", help="Path to the candidate runtime_summary.json, e.g. Colab GPU run")
    return parser.parse_args()


def load_summary(path: str) -> dict:
    return json.loads(Path(path).read_text())


def workload_signature(summary: dict) -> tuple[int, int, int, int, int, tuple[str, ...]]:
    return (
        int(summary["train_runs"]),
        int(summary["eval_runs"]),
        int(summary["duration_seconds"]),
        int(summary["epochs"]),
        int(summary["batch_size"]),
        tuple(sorted(summary["models"].keys())),
    )


def safe_speedup(baseline_value: float, candidate_value: float) -> float:
    return baseline_value / max(candidate_value, 1e-9)


def main() -> None:
    args = parse_args()
    baseline = load_summary(args.baseline)
    candidate = load_summary(args.candidate)

    baseline_sig = workload_signature(baseline)
    candidate_sig = workload_signature(candidate)
    if baseline_sig != candidate_sig:
        raise SystemExit(
            "Runtime summaries describe different workloads. "
            "Make sure train/eval runs, duration, epochs, batch size, and models all match."
        )

    result = {
        "baseline_device": baseline["device"],
        "candidate_device": candidate["device"],
        "overall": {
            "baseline_total_wall_seconds": baseline["total_wall_seconds"],
            "candidate_total_wall_seconds": candidate["total_wall_seconds"],
            "speedup_x": round(safe_speedup(float(baseline["total_wall_seconds"]), float(candidate["total_wall_seconds"])), 3),
        },
        "data_prep": {
            "baseline_seconds": baseline["data_prep_seconds"],
            "candidate_seconds": candidate["data_prep_seconds"],
            "speedup_x": round(safe_speedup(float(baseline["data_prep_seconds"]), float(candidate["data_prep_seconds"])), 3),
        },
        "models": {},
    }

    for model_name in sorted(baseline["models"].keys()):
        base_model = baseline["models"][model_name]
        cand_model = candidate["models"][model_name]
        result["models"][model_name] = {
            "fit_seconds": {
                "baseline": base_model["fit_seconds"],
                "candidate": cand_model["fit_seconds"],
                "speedup_x": round(safe_speedup(float(base_model["fit_seconds"]), float(cand_model["fit_seconds"])), 3),
            },
            "eval_seconds": {
                "baseline": base_model["eval_seconds"],
                "candidate": cand_model["eval_seconds"],
                "speedup_x": round(safe_speedup(float(base_model["eval_seconds"]), float(cand_model["eval_seconds"])), 3),
            },
            "total_seconds": {
                "baseline": base_model["total_seconds"],
                "candidate": cand_model["total_seconds"],
                "speedup_x": round(safe_speedup(float(base_model["total_seconds"]), float(cand_model["total_seconds"])), 3),
            },
        }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
