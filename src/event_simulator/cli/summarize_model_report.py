from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize learned-model benchmark reports with condition leaderboard metrics")
    parser.add_argument("report", help="Path to model_comparison.json")
    parser.add_argument("--baseline-report", help="Optional second report to compare against")
    return parser.parse_args()


def model_summary(model_payload: dict) -> dict[str, float]:
    metrics = model_payload["metrics"]
    time_conditions = model_payload["long_horizon"]["time_conditions"]
    values = [condition_metrics["balanced_accuracy"] for horizon in time_conditions.values() for condition_metrics in horizon.values()]
    brier_values = [condition_metrics["brier"] for horizon in time_conditions.values() for condition_metrics in horizon.values()]
    log_loss_values = [condition_metrics["log_loss"] for horizon in time_conditions.values() for condition_metrics in horizon.values()]
    return {
        "type_accuracy": float(metrics["type_accuracy"]),
        "family_accuracy": float(metrics["family_accuracy"]),
        "time_mae": float(metrics["time_mae"]),
        "mean_condition_balanced_accuracy": sum(values) / len(values),
        "mean_condition_brier": sum(brier_values) / len(brier_values),
        "mean_condition_log_loss": sum(log_loss_values) / len(log_loss_values),
    }


def load_report(path_str: str) -> dict:
    return json.loads(Path(path_str).read_text())


def print_report(report: dict, title: str) -> None:
    print(title)
    print(f"  train_runs={report['train_runs']} eval_runs={report['eval_runs']} duration={report['duration_seconds']}s")
    rows = []
    for name, payload in report["models"].items():
        summary = model_summary(payload)
        rows.append((name, summary))
    rows.sort(key=lambda item: item[1]["mean_condition_balanced_accuracy"], reverse=True)
    for name, summary in rows:
        print(
            f"  {name}: cond_bal_acc={summary['mean_condition_balanced_accuracy']:.4f} "
            f"brier={summary['mean_condition_brier']:.4f} log_loss={summary['mean_condition_log_loss']:.4f} "
            f"type_acc={summary['type_accuracy']:.4f} family_acc={summary['family_accuracy']:.4f} time_mae={summary['time_mae']:.4f}"
        )


def print_delta(current: dict, baseline: dict) -> None:
    print("Comparison vs baseline")
    for name, payload in current["models"].items():
        if name not in baseline["models"]:
            continue
        current_summary = model_summary(payload)
        baseline_summary = model_summary(baseline["models"][name])
        print(
            f"  {name}: "
            f"cond_bal_acc={current_summary['mean_condition_balanced_accuracy'] - baseline_summary['mean_condition_balanced_accuracy']:+.4f} "
            f"brier={current_summary['mean_condition_brier'] - baseline_summary['mean_condition_brier']:+.4f} "
            f"log_loss={current_summary['mean_condition_log_loss'] - baseline_summary['mean_condition_log_loss']:+.4f} "
            f"type_acc={current_summary['type_accuracy'] - baseline_summary['type_accuracy']:+.4f} "
            f"family_acc={current_summary['family_accuracy'] - baseline_summary['family_accuracy']:+.4f} "
            f"time_mae={current_summary['time_mae'] - baseline_summary['time_mae']:+.4f}"
        )


def main() -> None:
    args = parse_args()
    report = load_report(args.report)
    print_report(report, "Report summary")
    if args.baseline_report:
        baseline = load_report(args.baseline_report)
        print()
        print_delta(report, baseline)


if __name__ == "__main__":
    main()
