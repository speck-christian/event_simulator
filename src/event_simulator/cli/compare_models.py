from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

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
    NeuroSymbolicTPPBaseline,
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
        help="Comma-separated model names to run, or 'all'. Options: global_rate,transition,mechanistic,neural_tpp,multitask_neural_tpp,continuous_tpp,transformer_tpp,neuro_symbolic_tpp",
    )
    parser.add_argument(
        "--skip-prediction-dashboard",
        action="store_true",
        help="Skip building traffic_predictions.html. Useful when you only need the benchmark report and want to avoid heavy post-processing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overall_start = time.perf_counter()
    data_start = time.perf_counter()
    print(
        f"[compare_models] loading runs: train={args.train_runs} eval={args.eval_runs} "
        f"duration={args.duration}s control={args.control_mode} profile={args.simulation_profile}",
        flush=True,
    )
    cache_dir = args.cache_dir or f"analysis/cache_{args.control_mode}_{args.simulation_profile}"
    runs = load_or_generate_runs(
        args.train_runs + args.eval_runs,
        args.duration,
        args.seed_start,
        cache_dir,
        control_mode=args.control_mode,
        simulation_profile=args.simulation_profile,
    )
    data_seconds = time.perf_counter() - data_start
    print(f"[compare_models] runs ready from {cache_dir}", flush=True)
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
        "neuro_symbolic_tpp": NeuroSymbolicTPPBaseline(context_len=args.context_len, epochs=args.epochs, device=args.device),
    }
    if args.models == "all":
        selected_names = list(available_models.keys())
    else:
        selected_names = [name.strip() for name in args.models.split(",") if name.strip()]
        unknown = [name for name in selected_names if name not in available_models]
        if unknown:
            raise SystemExit(f"Unknown model names: {', '.join(unknown)}")

    model_outputs: dict[str, dict] = {}
    timing_outputs: dict[str, dict[str, float]] = {}
    for name in selected_names:
        model = available_models[name]
        print(f"[compare_models] fitting {name}", flush=True)
        fit_start = time.perf_counter()
        model.fit(train_runs)
        fit_seconds = time.perf_counter() - fit_start
        print(f"[compare_models] finished fitting {name} in {fit_seconds:.1f}s", flush=True)
        print(f"[compare_models] evaluating {name}", flush=True)
        eval_start = time.perf_counter()
        metrics, example_predictions, long_horizon = evaluate_model(
            model,
            eval_runs,
            log_progress=True,
            progress_prefix=name,
        )
        eval_seconds = time.perf_counter() - eval_start
        print(f"[compare_models] finished evaluating {name}", flush=True)
        model_outputs[model.name] = {
            "description": model.description,
            "metrics": metrics,
            "example_predictions": example_predictions,
            "long_horizon": long_horizon,
        }
        timing_outputs[model.name] = {
            "fit_seconds": round(fit_seconds, 3),
            "eval_seconds": round(eval_seconds, 3),
            "total_seconds": round(fit_seconds + eval_seconds, 3),
        }

    report = build_report(train_runs, eval_runs, model_outputs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_dir / "model_comparison.html").write_text(build_dashboard_html(report))
    print(f"[compare_models] wrote aggregate comparison reports to {output_dir}", flush=True)
    checkpoint_manifest: dict[str, str] = {}
    for name in selected_names:
        model = available_models[name]
        if hasattr(model, "save_checkpoint") and model.name in {"neural_tpp", "multitask_neural_tpp", "continuous_tpp", "transformer_tpp", "neuro_symbolic_tpp"}:
            checkpoint_path = checkpoint_dir / f"{model.name}.pt"
            model.save_checkpoint(checkpoint_path)
            checkpoint_manifest[model.name] = str(checkpoint_path)
            print(f"[compare_models] saved checkpoint for {model.name}", flush=True)
    runtime_report = {
        "device": args.device,
        "train_runs": args.train_runs,
        "eval_runs": args.eval_runs,
        "duration_seconds": args.duration,
        "control_mode": args.control_mode,
        "simulation_profile": args.simulation_profile,
        "epochs": args.epochs,
        "context_len": args.context_len,
        "data_prep_seconds": round(data_seconds, 3),
        "total_wall_seconds": round(time.perf_counter() - overall_start, 3),
        "models": timing_outputs,
        "checkpoints": checkpoint_manifest,
    }
    (output_dir / "runtime_summary.json").write_text(json.dumps(runtime_report, indent=2) + "\n")
    print(f"[compare_models] wrote runtime summary to {output_dir / 'runtime_summary.json'}", flush=True)
    if args.skip_prediction_dashboard:
        print("[compare_models] skipped traffic_predictions.html by request", flush=True)
    else:
        try:
            dashboard_models = {name: available_models[name] for name in selected_names}
            (output_dir / "traffic_predictions.html").write_text(
                build_prediction_dashboard_html(
                    report,
                    eval_runs[0],
                    cached_runs=runs,
                    model_instances=dashboard_models,
                )
            )
            print(f"[compare_models] wrote traffic_predictions.html to {output_dir}", flush=True)
        except Exception:
            error_text = traceback.format_exc()
            (output_dir / "traffic_predictions_error.txt").write_text(error_text)
            print(
                f"[compare_models] prediction dashboard failed; wrote {output_dir / 'traffic_predictions_error.txt'}",
                flush=True,
            )
    print(json.dumps({name: output["metrics"] for name, output in model_outputs.items()}, indent=2))
