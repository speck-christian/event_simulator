from __future__ import annotations

import argparse
import json
from pathlib import Path

from event_simulator.evaluation import (
    build_dashboard_html,
    build_prediction_dashboard_html,
    build_report,
    evaluate_model,
    load_or_generate_runs,
)
from event_simulator.models import (
    ContinuousTPPBaseline,
    GlobalRateBaseline,
    MechanisticBaseline,
    MultitaskNeuralTPPBaseline,
    NeuralTPPBaseline,
    TransformerTPPBaseline,
    TransitionBaseline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare event-prediction models on the traffic simulator")
    parser.add_argument("--train-runs", type=int, default=24)
    parser.add_argument("--eval-runs", type=int, default=6)
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--output-dir", default="analysis")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=None, help="Directory to save learned-model checkpoints. Defaults to <output-dir>/checkpoints")
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--context-len", type=int, default=32)
    parser.add_argument(
        "--control-mode",
        choices=("adaptive", "fixed"),
        default="adaptive",
        help="Traffic control mode used when generating or loading simulator runs.",
    )
    parser.add_argument(
        "--simulation-profile",
        choices=("baseline", "richer"),
        default="richer",
        help="Simulator dynamics profile. 'richer' adds burstier arrivals and lane-specific service rates.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for learned models: auto, cpu, cuda, or mps",
    )
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model names to run, or 'all'. Options: global_rate,transition,mechanistic,neural_tpp,multitask_neural_tpp,continuous_tpp,transformer_tpp",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir or f"analysis/cache_{args.control_mode}_{args.simulation_profile}"
    runs = load_or_generate_runs(
        args.train_runs + args.eval_runs,
        args.duration,
        args.seed_start,
        cache_dir,
        control_mode=args.control_mode,
        simulation_profile=args.simulation_profile,
    )
    train_runs = runs[: args.train_runs]
    eval_runs = runs[args.train_runs :]
    available_models = {
        "global_rate": GlobalRateBaseline(),
        "transition": TransitionBaseline(),
        "mechanistic": MechanisticBaseline(),
        "neural_tpp": NeuralTPPBaseline(context_len=args.context_len, epochs=args.epochs, device=args.device),
        "multitask_neural_tpp": MultitaskNeuralTPPBaseline(context_len=args.context_len, epochs=args.epochs, device=args.device),
        "continuous_tpp": ContinuousTPPBaseline(context_len=args.context_len, epochs=args.epochs, device=args.device),
        "transformer_tpp": TransformerTPPBaseline(context_len=args.context_len, epochs=args.epochs, device=args.device),
    }
    if args.models == "all":
        selected_names = list(available_models.keys())
    else:
        selected_names = [name.strip() for name in args.models.split(",") if name.strip()]
        unknown = [name for name in selected_names if name not in available_models]
        if unknown:
            raise SystemExit(f"Unknown model names: {', '.join(unknown)}")

    model_outputs: dict[str, dict] = {}
    for name in selected_names:
        model = available_models[name]
        model.fit(train_runs)
        metrics, example_predictions, long_horizon = evaluate_model(model, eval_runs)
        model_outputs[model.name] = {
            "description": model.description,
            "metrics": metrics,
            "example_predictions": example_predictions,
            "long_horizon": long_horizon,
        }

    report = build_report(train_runs, eval_runs, model_outputs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_dir / "model_comparison.html").write_text(build_dashboard_html(report))
    (output_dir / "traffic_predictions.html").write_text(build_prediction_dashboard_html(report, eval_runs[0], cached_runs=eval_runs))
    for name in selected_names:
        model = available_models[name]
        if hasattr(model, "save_checkpoint") and model.name in {"neural_tpp", "multitask_neural_tpp", "continuous_tpp", "transformer_tpp"}:
            model.save_checkpoint(checkpoint_dir / f"{model.name}.pt")
    print(json.dumps({name: output["metrics"] for name, output in model_outputs.items()}, indent=2))
