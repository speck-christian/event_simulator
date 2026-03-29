from __future__ import annotations

import json
from typing import Any


CONDITION_DESCRIPTIONS = {
    "congested": "Mean balanced accuracy for predicting whether total queue reaches the congested threshold.",
    "severe_queue": "Mean balanced accuracy for predicting whether any single approach develops a severe queue.",
    "ns_pressure_high": "Mean balanced accuracy for predicting elevated north-south corridor pressure.",
    "ew_pressure_high": "Mean balanced accuracy for predicting elevated east-west corridor pressure.",
    "pressure_imbalance": "Mean balanced accuracy for predicting a strong imbalance between the two corridors.",
}


def _mean_condition_metric(report: dict[str, Any], model_name: str, metric_name: str) -> float:
    values: list[float] = []
    for horizon in ("10s", "30s", "60s"):
        for stats in report["models"][model_name]["long_horizon"]["time_conditions"][horizon].values():
            values.append(float(stats[metric_name]))
    return round(sum(values) / max(1, len(values)), 4)


def _mean_condition_metric_by_name(report: dict[str, Any], model_name: str, condition_name: str, metric_name: str) -> float:
    values: list[float] = []
    for horizon in ("10s", "30s", "60s"):
        values.append(float(report["models"][model_name]["long_horizon"]["time_conditions"][horizon][condition_name][metric_name]))
    return round(sum(values) / max(1, len(values)), 4)


def build_regime_report(fixed_report: dict[str, Any], adaptive_report: dict[str, Any]) -> dict[str, Any]:
    model_names = sorted(set(fixed_report["models"]).intersection(adaptive_report["models"]))
    models: dict[str, Any] = {}
    for name in model_names:
        fixed_metrics = fixed_report["models"][name]["metrics"]
        adaptive_metrics = adaptive_report["models"][name]["metrics"]
        models[name] = {
            "description": fixed_report["models"][name]["description"],
            "fixed": {
                "type_accuracy": fixed_metrics["type_accuracy"],
                "family_accuracy": fixed_metrics["family_accuracy"],
                "time_mae": fixed_metrics["time_mae"],
                "mean_condition_balanced_accuracy": _mean_condition_metric(fixed_report, name, "balanced_accuracy"),
                "mean_condition_brier": _mean_condition_metric(fixed_report, name, "brier"),
                "mean_condition_log_loss": _mean_condition_metric(fixed_report, name, "log_loss"),
            },
            "adaptive": {
                "type_accuracy": adaptive_metrics["type_accuracy"],
                "family_accuracy": adaptive_metrics["family_accuracy"],
                "time_mae": adaptive_metrics["time_mae"],
                "mean_condition_balanced_accuracy": _mean_condition_metric(adaptive_report, name, "balanced_accuracy"),
                "mean_condition_brier": _mean_condition_metric(adaptive_report, name, "brier"),
                "mean_condition_log_loss": _mean_condition_metric(adaptive_report, name, "log_loss"),
            },
            "per_condition": {
                condition_name: {
                    "fixed_mean_balanced_accuracy": _mean_condition_metric_by_name(
                        fixed_report, name, condition_name, "balanced_accuracy"
                    ),
                    "adaptive_mean_balanced_accuracy": _mean_condition_metric_by_name(
                        adaptive_report, name, condition_name, "balanced_accuracy"
                    ),
                }
                for condition_name in CONDITION_DESCRIPTIONS
            },
        }
    return {
        "fixed_source": {
            "train_runs": fixed_report["train_runs"],
            "eval_runs": fixed_report["eval_runs"],
            "duration_seconds": fixed_report["duration_seconds"],
        },
        "adaptive_source": {
            "train_runs": adaptive_report["train_runs"],
            "eval_runs": adaptive_report["eval_runs"],
            "duration_seconds": adaptive_report["duration_seconds"],
        },
        "condition_descriptions": CONDITION_DESCRIPTIONS,
        "models": models,
    }


def build_regime_dashboard_html(report: dict[str, Any]) -> str:
    payload = json.dumps(report).replace("</", "<\\/")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Fixed vs Adaptive Comparison</title>
  <style>
    :root {{
      --bg: #f4efe8;
      --panel: rgba(255,255,255,0.86);
      --ink: #1f2933;
      --muted: #56616b;
      --line: rgba(31,41,51,0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: linear-gradient(135deg, #e6dccb, var(--bg));
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
    }}
    .page {{
      width: min(1280px, calc(100vw - 32px));
      margin: 24px auto;
      display: grid;
      gap: 18px;
    }}
    .panel {{
      background: var(--panel);
      border-radius: 24px;
      border: 1px solid var(--line);
      box-shadow: 0 14px 36px rgba(55, 41, 24, 0.12);
      padding: 20px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 18px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
    }}
    .stat {{
      background: rgba(255,255,255,0.58);
      border-radius: 18px;
      padding: 14px;
    }}
    .stat strong {{
      display: block;
      font-size: 1.2rem;
    }}
    h1, h2 {{ margin: 0 0 12px; }}
    .lede {{ margin: 0; line-height: 1.55; color: var(--muted); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.95rem; }}
    th, td {{ padding: 10px 8px; border-bottom: 1px solid var(--line); text-align: left; }}
    .condition-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 14px;
      margin-top: 12px;
    }}
    .condition-card {{
      background: rgba(255,255,255,0.58);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
    }}
    .condition-card h3 {{ margin: 0 0 8px; font-size: 1rem; }}
    .condition-card p {{ margin: 0 0 10px; color: var(--muted); font-size: 0.9rem; line-height: 1.45; }}
    .chart-wrap {{ overflow-x: auto; }}
    svg {{
      width: 100%;
      height: auto;
      background: rgba(255,255,255,0.45);
      border-radius: 18px;
    }}
    .legend {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      font-size: 0.92rem;
      color: var(--muted);
    }}
    .legend span::before {{
      content: "";
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 8px;
      background: var(--color);
    }}
    @media (max-width: 980px) {{
      .hero {{ grid-template-columns: 1fr; }}
      .stats {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="panel hero">
      <div>
        <h1>Fixed vs Adaptive 60/20</h1>
        <p class="lede">This page compares the same model family across fixed-control and adaptive-control traffic regimes at the same benchmark scale. All condition comparisons here use mean balanced accuracy over the fixed-time condition task, so the regime differences stay on a single metric scale.</p>
      </div>
      <div class="stats">
        <div class="stat"><span>Fixed Benchmark</span><strong id="fixed-benchmark"></strong></div>
        <div class="stat"><span>Adaptive Benchmark</span><strong id="adaptive-benchmark"></strong></div>
      </div>
    </section>

    <section class="panel">
      <h2>Aggregate Regime Comparison</h2>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Fixed cond. bal. acc</th>
            <th>Adaptive cond. bal. acc</th>
            <th>Delta</th>
            <th>Fixed Brier</th>
            <th>Adaptive Brier</th>
            <th>Fixed Time MAE</th>
            <th>Adaptive Time MAE</th>
          </tr>
        </thead>
        <tbody id="regime-body"></tbody>
      </table>
    </section>

    <section class="panel">
      <h2>Per-Condition Fixed vs Adaptive</h2>
      <p class="lede">Each card compares the same per-condition balanced-accuracy metric under fixed and adaptive control. Bars are paired by model, so positive gaps indicate conditions where adaptive control helps that model more than fixed control.</p>
      <div class="condition-grid" id="condition-grid"></div>
    </section>
  </div>

  <script id="payload" type="application/json">{payload}</script>
  <script>
    const report = JSON.parse(document.getElementById("payload").textContent);
    const modelNames = Object.keys(report.models);
    const colors = {{
      global_rate: "#7a4f9a",
      transition: "#c75c38",
      mechanistic: "#2f7d5d",
      neural_tpp: "#2e5bff",
      multitask_neural_tpp: "#b83280",
      continuous_tpp: "#0f8b8d",
      transformer_tpp: "#d97706"
    }};
    const shortNames = {{
      global_rate: "Global",
      transition: "Transition",
      mechanistic: "Mechanistic",
      neural_tpp: "Neural",
      multitask_neural_tpp: "MT Neural",
      continuous_tpp: "Continuous",
      transformer_tpp: "Transformer"
    }};

    document.getElementById("fixed-benchmark").textContent = `${{report.fixed_source.train_runs}} / ${{report.fixed_source.eval_runs}} / ${{report.fixed_source.duration_seconds}}s`;
    document.getElementById("adaptive-benchmark").textContent = `${{report.adaptive_source.train_runs}} / ${{report.adaptive_source.eval_runs}} / ${{report.adaptive_source.duration_seconds}}s`;

    const regimeBody = document.getElementById("regime-body");
    modelNames
      .map((name) => [name, report.models[name]])
      .sort((a, b) => b[1].adaptive.mean_condition_balanced_accuracy - a[1].adaptive.mean_condition_balanced_accuracy)
      .forEach(([name, model]) => {{
        const row = document.createElement("tr");
        const delta = model.adaptive.mean_condition_balanced_accuracy - model.fixed.mean_condition_balanced_accuracy;
        row.innerHTML = `
          <td><strong>${{name}}</strong><br /><span style="color:#56616b">${{model.description}}</span></td>
          <td>${{model.fixed.mean_condition_balanced_accuracy.toFixed(4)}}</td>
          <td>${{model.adaptive.mean_condition_balanced_accuracy.toFixed(4)}}</td>
          <td>${{delta >= 0 ? "+" : ""}}${{delta.toFixed(4)}}</td>
          <td>${{model.fixed.mean_condition_brier.toFixed(4)}}</td>
          <td>${{model.adaptive.mean_condition_brier.toFixed(4)}}</td>
          <td>${{model.fixed.time_mae.toFixed(4)}}</td>
          <td>${{model.adaptive.time_mae.toFixed(4)}}</td>
        `;
        regimeBody.appendChild(row);
      }});

    function buildConditionCardSvg(conditionName) {{
      const items = modelNames.map((name) => ({{
        name,
        fixed: report.models[name].per_condition[conditionName].fixed_mean_balanced_accuracy,
        adaptive: report.models[name].per_condition[conditionName].adaptive_mean_balanced_accuracy,
      }}));
      const width = 420;
      const height = 320;
      const margin = {{ top: 24, right: 18, bottom: 66, left: 46 }};
      const innerWidth = width - margin.left - margin.right;
      const innerHeight = height - margin.top - margin.bottom;
      const maxValue = Math.max(0.6, ...items.flatMap((item) => [item.fixed, item.adaptive])) * 1.05;
      const minValue = Math.max(0.0, Math.min(...items.flatMap((item) => [item.fixed, item.adaptive])) - 0.08);
      const range = Math.max(0.05, maxValue - minValue);
      const step = innerWidth / Math.max(items.length, 1);
      const barWidth = Math.min(16, step * 0.24);
      const yFor = (value) => margin.top + innerHeight - innerHeight * ((value - minValue) / range);
      const ticks = [0, 0.25, 0.5, 0.75, 1.0].map((ratio) => {{
        const tick = minValue + range * ratio;
        const y = yFor(tick);
        return `
          <line x1="${{margin.left}}" y1="${{y.toFixed(1)}}" x2="${{(width - margin.right).toFixed(1)}}" y2="${{y.toFixed(1)}}" stroke="rgba(31,41,51,0.08)" />
          <text x="${{(margin.left - 8).toFixed(1)}}" y="${{(y + 4).toFixed(1)}}" text-anchor="end" font-size="10" fill="#56616b">${{tick.toFixed(3)}}</text>
        `;
      }}).join("");
      const bars = items.map((item, index) => {{
        const center = margin.left + step * (index + 0.5);
        const fixedHeight = innerHeight * ((item.fixed - minValue) / range);
        const adaptiveHeight = innerHeight * ((item.adaptive - minValue) / range);
        return `
          <rect x="${{(center - barWidth - 3).toFixed(1)}}" y="${{(margin.top + innerHeight - fixedHeight).toFixed(1)}}" width="${{barWidth.toFixed(1)}}" height="${{fixedHeight.toFixed(1)}}" rx="4" fill="${{colors[item.name] || "#444"}}" fill-opacity="0.9" />
          <rect x="${{(center + 3).toFixed(1)}}" y="${{(margin.top + innerHeight - adaptiveHeight).toFixed(1)}}" width="${{barWidth.toFixed(1)}}" height="${{adaptiveHeight.toFixed(1)}}" rx="4" fill="${{colors[item.name] || "#444"}}" fill-opacity="0.35" stroke="${{colors[item.name] || "#444"}}" stroke-width="1.2" />
          <text x="${{center.toFixed(1)}}" y="${{(height - 20).toFixed(1)}}" text-anchor="middle" font-size="10" fill="#56616b">${{shortNames[item.name] || item.name}}</text>
        `;
      }}).join("");
      return `
        <svg viewBox="0 0 420 320" preserveAspectRatio="none">
          ${{ticks}}
          <line x1="${{margin.left}}" y1="${{(margin.top + innerHeight).toFixed(1)}}" x2="${{(width - margin.right).toFixed(1)}}" y2="${{(margin.top + innerHeight).toFixed(1)}}" stroke="rgba(31,41,51,0.16)" />
          <text x="${{margin.left}}" y="${{(margin.top - 8).toFixed(1)}}" font-size="11" fill="#56616b">Mean Balanced Accuracy</text>
          ${{bars}}
          <text x="${{width / 2}}" y="${{(height - 4).toFixed(1)}}" text-anchor="middle" font-size="10" fill="#56616b">solid = fixed, faded = adaptive</text>
        </svg>
      `;
    }}

    const conditionGrid = document.getElementById("condition-grid");
    Object.entries(report.condition_descriptions).forEach(([conditionName, description]) => {{
      const deltas = modelNames.map((name) => {{
        const item = report.models[name].per_condition[conditionName];
        return {{
          name,
          delta: item.adaptive_mean_balanced_accuracy - item.fixed_mean_balanced_accuracy,
        }};
      }}).sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
      const topDelta = deltas[0];
      const card = document.createElement("div");
      card.className = "condition-card";
      card.innerHTML = `
        <h3>${{conditionName}}</h3>
        <p>${{description}}</p>
        <p><strong>Largest regime shift:</strong> ${{shortNames[topDelta.name] || topDelta.name}} (${{topDelta.delta >= 0 ? "+" : ""}}${{topDelta.delta.toFixed(4)}} adaptive - fixed).</p>
        ${{buildConditionCardSvg(conditionName)}}
      `;
      conditionGrid.appendChild(card);
    }});
  </script>
</body>
</html>
"""
