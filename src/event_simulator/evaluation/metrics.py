from __future__ import annotations

from collections import defaultdict
import math
import statistics
from typing import Any

from event_simulator.models.base import Predictor
from event_simulator.models.common import condition_flags, event_family, event_label, mean_or_default, rollout_predicted_events
from event_simulator.models.common.replay import ReplayState, make_synthetic_event


def init_condition_stats() -> dict[str, dict[str, int]]:
    return {
        name: {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "count": 0, "brier_sum": 0.0, "log_loss_sum": 0.0, "score_count": 0}
        for name in condition_flags(ReplayState())
    }


def init_calibration_bins(num_bins: int = 5) -> list[dict[str, float]]:
    return [{"count": 0, "score_sum": 0.0, "actual_sum": 0.0} for _ in range(num_bins)]


def init_condition_calibration_bins(num_bins: int = 5) -> dict[str, list[dict[str, float]]]:
    return {
        name: init_calibration_bins(num_bins)
        for name in condition_flags(ReplayState())
    }


def accumulate_calibration_bins(
    bins: list[dict[str, float]],
    predicted_scores: dict[str, float],
    actual_flags: dict[str, bool],
) -> None:
    for name, score in predicted_scores.items():
        clipped = min(1.0, max(0.0, float(score)))
        index = min(len(bins) - 1, int(clipped * len(bins)))
        bins[index]["count"] += 1
        bins[index]["score_sum"] += clipped
        bins[index]["actual_sum"] += 1.0 if actual_flags[name] else 0.0


def accumulate_condition_calibration_bins(
    bins_by_condition: dict[str, list[dict[str, float]]],
    predicted_scores: dict[str, float],
    actual_flags: dict[str, bool],
) -> None:
    for name, bins in bins_by_condition.items():
        if name not in predicted_scores:
            continue
        clipped = min(1.0, max(0.0, float(predicted_scores[name])))
        index = min(len(bins) - 1, int(clipped * len(bins)))
        bins[index]["count"] += 1
        bins[index]["score_sum"] += clipped
        bins[index]["actual_sum"] += 1.0 if actual_flags[name] else 0.0


def finalize_calibration_bins(bins: list[dict[str, float]]) -> list[dict[str, float]]:
    finalized: list[dict[str, float]] = []
    for index, bucket in enumerate(bins):
        count = int(bucket["count"])
        finalized.append(
            {
                "bin_start": round(index / len(bins), 2),
                "bin_end": round((index + 1) / len(bins), 2),
                "count": count,
                "mean_score": round(bucket["score_sum"] / max(1, count), 4),
                "actual_rate": round(bucket["actual_sum"] / max(1, count), 4),
            }
        )
    return finalized


def summarize_calibration_bins(bins: list[dict[str, float]]) -> dict[str, float]:
    total = sum(int(bucket["count"]) for bucket in bins)
    if total <= 0:
        return {
            "ece": 0.0,
            "mce": 0.0,
            "mean_confidence": 0.0,
            "mean_observed_rate": 0.0,
            "support": 0,
        }
    weighted_gap = 0.0
    max_gap = 0.0
    score_sum = 0.0
    actual_sum = 0.0
    for bucket in bins:
        count = int(bucket["count"])
        if count <= 0:
            continue
        mean_score = float(bucket["score_sum"]) / count
        actual_rate = float(bucket["actual_sum"]) / count
        gap = abs(mean_score - actual_rate)
        weighted_gap += gap * count
        max_gap = max(max_gap, gap)
        score_sum += mean_score * count
        actual_sum += actual_rate * count
    return {
        "ece": round(weighted_gap / total, 4),
        "mce": round(max_gap, 4),
        "mean_confidence": round(score_sum / total, 4),
        "mean_observed_rate": round(actual_sum / total, 4),
        "support": total,
    }


def finalize_condition_calibration_bins(
    bins_by_condition: dict[str, list[dict[str, float]]],
) -> dict[str, list[dict[str, float]]]:
    return {
        name: finalize_calibration_bins(bins)
        for name, bins in bins_by_condition.items()
    }


def summarize_condition_calibration_bins(
    bins_by_condition: dict[str, list[dict[str, float]]],
) -> dict[str, dict[str, float]]:
    return {
        name: summarize_calibration_bins(bins)
        for name, bins in bins_by_condition.items()
    }


def accumulate_condition_stats(
    stats: dict[str, dict[str, int]],
    predicted_flags: dict[str, bool],
    actual_flags: dict[str, bool],
    predicted_scores: dict[str, float] | None = None,
) -> None:
    for name in stats:
        predicted = bool(predicted_flags[name])
        actual = bool(actual_flags[name])
        stats[name]["count"] += 1
        if predicted and actual:
            stats[name]["tp"] += 1
        elif predicted and not actual:
            stats[name]["fp"] += 1
        elif not predicted and actual:
            stats[name]["fn"] += 1
        else:
            stats[name]["tn"] += 1
        if predicted_scores is not None and name in predicted_scores:
            score = min(1.0, max(0.0, float(predicted_scores[name])))
            target = 1.0 if actual else 0.0
            stats[name]["brier_sum"] += (score - target) ** 2
            stats[name]["log_loss_sum"] += -(target * math.log(max(score, 1e-6)) + (1.0 - target) * math.log(max(1e-6, 1.0 - score)))
            stats[name]["score_count"] += 1


def finalize_condition_stats(stats: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    finalized: dict[str, dict[str, float]] = {}
    for name, values in stats.items():
        count = max(1, values["count"])
        tp = values["tp"]
        tn = values["tn"]
        fp = values["fp"]
        fn = values["fn"]
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        specificity = tn / max(1, tn + fp)
        balanced_accuracy = 0.5 * (recall + specificity)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        finalized[name] = {
            "accuracy": round((tp + tn) / count, 4),
            "balanced_accuracy": round(balanced_accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "brier": round(values["brier_sum"] / max(1, values["score_count"]), 4),
            "log_loss": round(values["log_loss_sum"] / max(1, values["score_count"]), 4),
            "actual_positive_rate": round((tp + fn) / count, 4),
            "predicted_positive_rate": round((tp + fp) / count, 4),
            "comparisons": values["count"],
            "score_comparisons": values["score_count"],
        }
    return finalized


def rollout_state_after_prefix(
    base_state: ReplayState, prefix: list[Any], summary: dict[str, Any], predicted: bool
) -> ReplayState:
    rolled_state = base_state.clone()
    for item in prefix:
        if predicted:
            event = make_synthetic_event(item["label"], float(item["time_s"]), rolled_state)
        else:
            event = item
        rolled_state.update(event, summary)
    return rolled_state


def rollout_predicted_state_until_time(
    model: Predictor, base_state: ReplayState, summary: dict[str, Any], target_time: float, max_steps: int = 128
) -> ReplayState:
    rolled_state = base_state.clone()
    steps = 0
    while rolled_state.current_time < target_time and steps < max_steps:
        predicted_label, predicted_time = model.predict(rolled_state, summary)
        predicted_time = max(predicted_time, rolled_state.current_time + 0.01)
        if predicted_time > target_time:
            break
        event = make_synthetic_event(predicted_label, predicted_time, rolled_state)
        rolled_state.update(event, summary)
        steps += 1
    rolled_state.current_time = max(rolled_state.current_time, target_time)
    return rolled_state


def actual_state_until_time(
    base_state: ReplayState, future_events: list[dict[str, Any]], summary: dict[str, Any], target_time: float
) -> ReplayState:
    rolled_state = base_state.clone()
    for event in future_events:
        if float(event["time_s"]) > target_time:
            break
        rolled_state.update(event, summary)
    rolled_state.current_time = max(rolled_state.current_time, target_time)
    return rolled_state


def evaluate_model(model: Predictor, eval_runs: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    all_predictions: list[dict[str, Any]] = []
    example_predictions: list[dict[str, Any]] = []
    example_time_condition_predictions: list[dict[str, Any]] = []
    rollout_horizons = (5, 10)
    rollout_stats = {
        horizon: {"type_correct": 0, "family_correct": 0, "time_abs_error": [], "count": 0}
        for horizon in rollout_horizons
    }
    condition_stats = {horizon: init_condition_stats() for horizon in rollout_horizons}
    time_condition_horizons = (10.0, 30.0, 60.0)
    time_condition_stats = {horizon: init_condition_stats() for horizon in time_condition_horizons}
    calibration_bins = init_calibration_bins()
    condition_calibration_bins = init_condition_calibration_bins()
    direct_time_condition_horizons = list(time_condition_horizons)
    long_horizon_stride = 10
    family_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"type_correct": 0, "family_correct": 0, "time_abs_error": [], "count": 0}
    )
    rollout_family_stats: dict[int, dict[str, dict[str, Any]]] = {
        horizon: defaultdict(lambda: {"type_correct": 0, "family_correct": 0, "time_abs_error": [], "count": 0})
        for horizon in rollout_horizons
    }

    for run_index, run in enumerate(eval_runs):
        state = ReplayState()
        events = run["events"]
        for event_index, (current_event, actual_next_event) in enumerate(zip(events, events[1:])):
            state.update(current_event, run["summary"])
            predicted_label, predicted_time = model.predict(state, run["summary"])
            actual_label = event_label(actual_next_event)
            actual_time = float(actual_next_event["time_s"])
            actual_family = event_family(actual_label)
            record = {
                "run_index": run_index,
                "context_time": state.current_time,
                "predicted_label": predicted_label,
                "predicted_time": round(predicted_time, 3),
                "actual_label": actual_label,
                "actual_time": round(actual_time, 3),
                "abs_time_error": round(abs(predicted_time - actual_time), 3),
                "label_correct": predicted_label == actual_label,
                "family_correct": event_family(predicted_label) == actual_family,
            }
            all_predictions.append(record)
            if run_index == 0:
                example_predictions.append(record)
            family_stats[actual_family]["type_correct"] += int(record["label_correct"])
            family_stats[actual_family]["family_correct"] += int(record["family_correct"])
            family_stats[actual_family]["time_abs_error"].append(record["abs_time_error"])
            family_stats[actual_family]["count"] += 1

            future_events = events[event_index + 1 :]
            direct_predictions = model.predict_time_conditions(state, run["summary"], direct_time_condition_horizons)
            direct_scores = model.predict_time_condition_scores(state, run["summary"], direct_time_condition_horizons)
            if run_index == 0:
                example_item = {
                    "context_time": round(float(state.current_time), 3),
                    "horizons": {},
                }
                for time_horizon in time_condition_horizons:
                    target_time = state.current_time + time_horizon
                    horizon_key = f"{int(time_horizon)}s"
                    if target_time > float(run["summary"]["duration_seconds"]):
                        example_item["horizons"][horizon_key] = {
                            "flags": {name: None for name in condition_flags(ReplayState())},
                            "scores": {name: None for name in condition_flags(ReplayState())},
                        }
                        continue
                    predicted_flags = direct_predictions.get(horizon_key) if direct_predictions is not None else None
                    predicted_score_map = direct_scores.get(horizon_key) if direct_scores is not None else None
                    if predicted_flags is None:
                        predicted_state = rollout_predicted_state_until_time(model, state, run["summary"], target_time)
                        predicted_flags = condition_flags(predicted_state)
                    if predicted_score_map is None:
                        predicted_score_map = {name: 1.0 if predicted_flags[name] else 0.0 for name in predicted_flags}
                    example_item["horizons"][horizon_key] = {
                        "flags": {name: bool(predicted_flags[name]) for name in predicted_flags},
                        "scores": {name: round(float(predicted_score_map[name]), 4) for name in predicted_score_map},
                    }
                example_time_condition_predictions.append(example_item)

            if event_index % long_horizon_stride != 0:
                continue

            max_horizon = max(rollout_horizons)
            predicted_prefix = rollout_predicted_events(model, state, run["summary"], max_horizon)
            for horizon in rollout_horizons:
                actual_prefix = future_events[:horizon]
                if len(actual_prefix) < horizon:
                    continue
                for predicted_item, actual_item in zip(predicted_prefix[:horizon], actual_prefix):
                    actual_prefix_label = event_label(actual_item)
                    actual_prefix_family = event_family(actual_prefix_label)
                    rollout_stats[horizon]["type_correct"] += int(predicted_item["label"] == actual_prefix_label)
                    rollout_stats[horizon]["family_correct"] += int(
                        event_family(predicted_item["label"]) == event_family(actual_prefix_label)
                    )
                    time_abs_error = abs(predicted_item["time_s"] - float(actual_item["time_s"]))
                    rollout_stats[horizon]["time_abs_error"].append(time_abs_error)
                    rollout_stats[horizon]["count"] += 1
                    rollout_family_stats[horizon][actual_prefix_family]["type_correct"] += int(
                        predicted_item["label"] == actual_prefix_label
                    )
                    rollout_family_stats[horizon][actual_prefix_family]["family_correct"] += int(
                        event_family(predicted_item["label"]) == actual_prefix_family
                    )
                    rollout_family_stats[horizon][actual_prefix_family]["time_abs_error"].append(time_abs_error)
                    rollout_family_stats[horizon][actual_prefix_family]["count"] += 1
                predicted_state = rollout_state_after_prefix(state, predicted_prefix[:horizon], run["summary"], predicted=True)
                actual_state = rollout_state_after_prefix(state, actual_prefix, run["summary"], predicted=False)
                accumulate_condition_stats(
                    condition_stats[horizon],
                    condition_flags(predicted_state),
                    condition_flags(actual_state),
                )
            for time_horizon in time_condition_horizons:
                target_time = state.current_time + time_horizon
                if target_time > float(run["summary"]["duration_seconds"]):
                    continue
                actual_state = actual_state_until_time(state, future_events, run["summary"], target_time)
                if direct_predictions is not None:
                    predicted_flags = direct_predictions.get(f"{int(time_horizon)}s")
                else:
                    predicted_flags = None
                if direct_scores is not None:
                    predicted_score_map = direct_scores.get(f"{int(time_horizon)}s")
                else:
                    predicted_score_map = None
                if predicted_flags is None:
                    predicted_state = rollout_predicted_state_until_time(model, state, run["summary"], target_time)
                    predicted_flags = condition_flags(predicted_state)
                if predicted_score_map is None:
                    predicted_score_map = {name: 1.0 if predicted_flags[name] else 0.0 for name in predicted_flags}
                accumulate_condition_stats(
                    time_condition_stats[time_horizon],
                    predicted_flags,
                    condition_flags(actual_state),
                    predicted_score_map,
                )
                accumulate_calibration_bins(calibration_bins, predicted_score_map, condition_flags(actual_state))
                accumulate_condition_calibration_bins(condition_calibration_bins, predicted_score_map, condition_flags(actual_state))

    metrics = {
        "type_accuracy": round(sum(item["label_correct"] for item in all_predictions) / len(all_predictions), 4),
        "family_accuracy": round(sum(item["family_correct"] for item in all_predictions) / len(all_predictions), 4),
        "time_mae": round(statistics.fmean(item["abs_time_error"] for item in all_predictions), 4),
        "time_rmse": round(math.sqrt(statistics.fmean(item["abs_time_error"] ** 2 for item in all_predictions)), 4),
        "within_2s": round(sum(item["abs_time_error"] <= 2.0 for item in all_predictions) / len(all_predictions), 4),
        "predictions": len(all_predictions),
    }
    per_family_metrics = {
        family: {
            "type_accuracy": round(stats["type_correct"] / max(1, stats["count"]), 4),
            "family_accuracy": round(stats["family_correct"] / max(1, stats["count"]), 4),
            "time_mae": round(mean_or_default(stats["time_abs_error"], 0.0), 4),
            "predictions": stats["count"],
        }
        for family, stats in sorted(family_stats.items())
    }
    rollout_metrics = {
        str(horizon): {
            "type_accuracy": round(rollout_stats[horizon]["type_correct"] / max(1, rollout_stats[horizon]["count"]), 4),
            "family_accuracy": round(rollout_stats[horizon]["family_correct"] / max(1, rollout_stats[horizon]["count"]), 4),
            "time_mae": round(mean_or_default(rollout_stats[horizon]["time_abs_error"], 0.0), 4),
            "comparisons": rollout_stats[horizon]["count"],
            "per_family": {
                family: {
                    "type_accuracy": round(stats["type_correct"] / max(1, stats["count"]), 4),
                    "family_accuracy": round(stats["family_correct"] / max(1, stats["count"]), 4),
                    "time_mae": round(mean_or_default(stats["time_abs_error"], 0.0), 4),
                    "comparisons": stats["count"],
                }
                for family, stats in sorted(rollout_family_stats[horizon].items())
            },
            "conditions": finalize_condition_stats(condition_stats[horizon]),
        }
        for horizon in rollout_horizons
    }
    time_condition_metrics = {
        f"{int(horizon)}s": finalize_condition_stats(time_condition_stats[horizon])
        for horizon in time_condition_horizons
    }
    metrics["per_family"] = per_family_metrics
    return metrics, example_predictions, {
        "rollout": rollout_metrics,
        "time_conditions": time_condition_metrics,
        "time_condition_calibration": finalize_calibration_bins(calibration_bins),
        "time_condition_calibration_summary": summarize_calibration_bins(calibration_bins),
        "time_condition_calibration_by_condition": finalize_condition_calibration_bins(condition_calibration_bins),
        "time_condition_calibration_summary_by_condition": summarize_condition_calibration_bins(condition_calibration_bins),
        "example_time_condition_predictions": example_time_condition_predictions,
    }
