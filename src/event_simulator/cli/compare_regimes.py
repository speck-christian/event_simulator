from __future__ import annotations

import argparse
import json
from pathlib import Path

from event_simulator.evaluation.regime_dashboard import build_regime_dashboard_html, build_regime_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare fixed and adaptive benchmark reports")
    parser.add_argument("fixed_report")
    parser.add_argument("adaptive_report")
    parser.add_argument("--output-dir", default="analysis/regime_comparison_60_20")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixed_report = json.loads(Path(args.fixed_report).read_text())
    adaptive_report = json.loads(Path(args.adaptive_report).read_text())
    report = build_regime_report(fixed_report, adaptive_report)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "regime_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_dir / "regime_comparison.html").write_text(build_regime_dashboard_html(report))


if __name__ == "__main__":
    main()
