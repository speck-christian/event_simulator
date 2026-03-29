from __future__ import annotations

import json
from typing import Any


def build_dashboard_html(report: dict[str, Any]) -> str:
    payload = json.dumps(report).replace("</", "<\\/")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Model Comparison Dashboard</title>
  <style>
    :root {{
      --bg: #f3efe5;
      --panel: rgba(255,255,255,0.82);
      --ink: #1f2933;
      --muted: #56616b;
      --line: rgba(31,41,51,0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: linear-gradient(135deg, #e9ddc8, var(--bg));
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
    h1, h2 {{
      margin: 0 0 12px;
    }}
    .lede {{
      margin: 0;
      line-height: 1.55;
      color: var(--muted);
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
    }}
    .stat {{
      background: rgba(255,255,255,0.58);
      border-radius: 18px;
      padding: 14px;
    }}
    .stat strong {{
      display: block;
      font-size: 1.4rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
    }}
    th, td {{
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      text-align: left;
    }}
    .family-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 14px;
      margin-top: 12px;
    }}
    .family-card {{
      background: rgba(255,255,255,0.58);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
    }}
    .family-card h3 {{
      margin: 0 0 10px;
      font-size: 1rem;
    }}
    .family-card p {{
      margin: 0 0 10px;
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.45;
    }}
    .legend {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      font-size: 0.92rem;
      color: var(--muted);
    }}
    .legend.is-interactive {{
      gap: 10px;
    }}
    .legend button {{
      appearance: none;
      border: 1px solid rgba(31,41,51,0.12);
      background: rgba(255,255,255,0.74);
      color: var(--ink);
      padding: 8px 12px;
      border-radius: 999px;
      cursor: pointer;
      font: inherit;
      transition: transform 120ms ease, opacity 120ms ease, border-color 120ms ease;
    }}
    .legend button:hover {{
      transform: translateY(-1px);
      border-color: rgba(31,41,51,0.24);
    }}
    .legend button.is-off {{
      opacity: 0.45;
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
    .chart-wrap {{
      overflow-x: auto;
    }}
    svg {{
      width: 100%;
      height: auto;
      background: rgba(255,255,255,0.45);
      border-radius: 18px;
    }}
    .family-card svg {{
      min-width: 0;
      background: rgba(255,255,255,0.72);
      border: 1px solid var(--line);
    }}
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
    .condition-card h3 {{
      margin: 0 0 8px;
      font-size: 1rem;
    }}
    .condition-card p {{
      margin: 0 0 10px;
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.45;
    }}
    .condition-card svg {{
      min-width: 0;
      background: rgba(255,255,255,0.72);
      border: 1px solid var(--line);
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
        <h1>Event Prediction Comparison</h1>
        <p class="lede">This dashboard compares next-event predictors on top of the traffic-intersection simulator. Each model observes the true history up to time <em>t</em>, predicts the next event label and time, and is then scored against the actual next event.</p>
      </div>
      <div class="stats">
        <div class="stat"><span>Train runs</span><strong id="train-runs"></strong></div>
        <div class="stat"><span>Eval runs</span><strong id="eval-runs"></strong></div>
        <div class="stat"><span>Example seed</span><strong id="example-seed"></strong></div>
      </div>
    </section>

    <section class="panel">
      <h2>Aggregate Metrics</h2>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Type accuracy</th>
            <th>Family accuracy</th>
            <th>Time MAE</th>
            <th>Time RMSE</th>
            <th>Within 2s</th>
            <th>Predictions</th>
          </tr>
        </thead>
        <tbody id="metric-body"></tbody>
      </table>
    </section>

    <section class="panel">
      <h2>Long-Horizon Event Rollout</h2>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>5-event type acc</th>
            <th>5-event family acc</th>
            <th>5-event time MAE</th>
            <th>10-event type acc</th>
            <th>10-event family acc</th>
            <th>10-event time MAE</th>
          </tr>
        </thead>
        <tbody id="rollout-body"></tbody>
      </table>
    </section>

    <section class="panel">
      <h2>Per-Family Performance</h2>
      <p class="lede">Each card breaks out one event family. The first row shows type accuracy, and the second row shows time MAE. Within each row, solid bars are one-step predictions and faded bars are 10-event rollout.</p>
      <div class="family-grid" id="family-grid"></div>
    </section>

    <section class="panel">
      <h2>Condition Forecasts</h2>
      <p class="lede">These plots compare higher-level traffic conditions derived from the rolled-out future state. Each card shows 10-event rollout balanced accuracy by model.</p>
      <div class="legend is-interactive" id="condition-legend"></div>
      <div class="condition-grid" id="condition-grid"></div>
    </section>

    <section class="panel">
      <h2>Time-Condition Forecasts</h2>
      <p class="lede">These plots score the same higher-level conditions at fixed future horizons instead of fixed event counts. Each card shows balanced accuracy at 10s, 30s, and 60s into the future.</p>
      <div class="legend is-interactive" id="time-condition-legend"></div>
      <div class="condition-grid" id="time-condition-grid"></div>
    </section>

    <section class="panel">
      <h2>Probabilistic Condition Metrics</h2>
      <p class="lede">These plots summarize the calibrated fixed-time condition probabilities. Lower Brier and log loss are better, and the cards below break those scores out by condition instead of only pooling everything together.</p>
      <div class="legend is-interactive" id="prob-condition-legend"></div>
      <div class="chart-wrap">
        <svg id="prob-condition-chart" viewBox="0 0 980 420" preserveAspectRatio="none"></svg>
      </div>
      <div class="condition-grid" id="prob-condition-grid"></div>
    </section>

    <section class="panel">
      <h2>Condition Calibration</h2>
      <p class="lede">These reliability curves compare each model's calibrated pooled condition probabilities against the observed event rate. The dashed diagonal is perfect calibration. The summary cards add numeric calibration error so you do not have to read everything from the curves by eye.</p>
      <div class="legend" id="calibration-legend"></div>
      <div class="condition-grid" id="calibration-summary-grid"></div>
      <div class="chart-wrap">
        <svg id="calibration-chart" viewBox="0 0 760 420" preserveAspectRatio="none"></svg>
      </div>
      <div class="condition-grid" id="calibration-grid"></div>
    </section>

    <section class="panel">
      <h2>Prediction vs Actual Next-Event Time</h2>
      <p class="lede">Black dots are the actual next-event times for the example run. Colored dots show each model's predicted next-event time from the same decision point, and the faint line shows the timing miss.</p>
      <div class="legend" id="timeline-legend"></div>
      <div class="chart-wrap">
        <svg id="timeline-chart" viewBox="0 0 1100 520" preserveAspectRatio="none"></svg>
      </div>
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
    const conditionDescriptions = {{
      congested: "Predicts whether the total queue across all approaches reaches a congested level within the rolled-out future.",
      severe_queue: "Predicts whether any single approach develops a notably large queue within the rolled-out future.",
      ns_pressure_high: "Predicts whether north-south demand dominates enough to create sustained pressure on that corridor.",
      ew_pressure_high: "Predicts whether east-west demand dominates enough to create sustained pressure on that corridor.",
      pressure_imbalance: "Predicts whether the pressure difference between the two corridors becomes materially unbalanced."
    }};

    document.getElementById("train-runs").textContent = report.train_runs;
    document.getElementById("eval-runs").textContent = report.eval_runs;
    document.getElementById("example-seed").textContent = report.example_seed;

    const metricBody = document.getElementById("metric-body");
    const rolloutBody = document.getElementById("rollout-body");
    const familyGrid = document.getElementById("family-grid");
    const conditionGrid = document.getElementById("condition-grid");
    const timeConditionGrid = document.getElementById("time-condition-grid");
    const probConditionGrid = document.getElementById("prob-condition-grid");
    const calibrationSummaryGrid = document.getElementById("calibration-summary-grid");
    const calibrationGrid = document.getElementById("calibration-grid");
    const activeConditionModels = new Set(modelNames);

    modelNames
      .map((name) => [name, report.models[name]])
      .sort((a, b) => b[1].metrics.type_accuracy - a[1].metrics.type_accuracy)
      .forEach(([name, model]) => {{
        const row = document.createElement("tr");
        row.innerHTML = `
          <td><strong>${{name}}</strong><br /><span style="color:#56616b">${{model.description}}</span></td>
          <td>${{model.metrics.type_accuracy}}</td>
          <td>${{model.metrics.family_accuracy}}</td>
          <td>${{model.metrics.time_mae}}</td>
          <td>${{model.metrics.time_rmse}}</td>
          <td>${{model.metrics.within_2s}}</td>
          <td>${{model.metrics.predictions}}</td>
        `;
        metricBody.appendChild(row);

        const rolloutRow = document.createElement("tr");
        rolloutRow.innerHTML = `
          <td><strong>${{name}}</strong></td>
          <td>${{model.long_horizon.rollout["5"].type_accuracy}}</td>
          <td>${{model.long_horizon.rollout["5"].family_accuracy}}</td>
          <td>${{model.long_horizon.rollout["5"].time_mae}}</td>
          <td>${{model.long_horizon.rollout["10"].type_accuracy}}</td>
          <td>${{model.long_horizon.rollout["10"].family_accuracy}}</td>
          <td>${{model.long_horizon.rollout["10"].time_mae}}</td>
        `;
        rolloutBody.appendChild(rolloutRow);
      }});

    function addLegend(containerId) {{
      const node = document.getElementById(containerId);
      modelNames.forEach((name) => {{
        const item = document.createElement("span");
        item.textContent = name;
        item.style.setProperty("--color", colors[name] || "#444");
        node.appendChild(item);
      }});
    }}

    addLegend("timeline-legend");
    addLegend("calibration-legend");

    function setupConditionLegend(containerId) {{
      const node = document.getElementById(containerId);
      node.innerHTML = "";
      modelNames.forEach((name) => {{
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = shortNames[name] || name;
        button.dataset.model = name;
        button.style.borderColor = colors[name] || "#444";
        button.style.boxShadow = `inset 0 0 0 1px ${{colors[name] || "#444"}}22`;
        button.addEventListener("click", () => {{
          if (activeConditionModels.has(name)) {{
            if (activeConditionModels.size === 1) {{
              activeConditionModels.clear();
              modelNames.forEach((modelName) => activeConditionModels.add(modelName));
            }} else {{
              activeConditionModels.delete(name);
            }}
          }} else {{
            activeConditionModels.add(name);
          }}
          syncConditionLegends();
          renderConditionCards();
          renderTimeConditionCards();
        }});
        node.appendChild(button);
      }});
    }}

    function syncConditionLegends() {{
      document.querySelectorAll("#condition-legend button, #time-condition-legend button, #prob-condition-legend button").forEach((button) => {{
        const name = button.dataset.model;
        const enabled = activeConditionModels.has(name);
        button.classList.toggle("is-off", !enabled);
        button.style.background = enabled ? "rgba(255,255,255,0.92)" : "rgba(255,255,255,0.44)";
      }});
    }}

    function buildFamilyChartSvg(items) {{
      const chartHeight = 620;
      const chartWidth = 420;
      const margin = {{ top: 28, right: 18, bottom: 78, left: 46 }};
      const innerWidth = chartWidth - margin.left - margin.right;
      const sectionGap = 64;
      const sectionHeight = (chartHeight - margin.top - margin.bottom - sectionGap) / 2;
      const accuracyTop = margin.top;
      const timeTop = margin.top + sectionHeight + sectionGap;
      const step = innerWidth / Math.max(items.length, 1);
      const barWidth = Math.min(18, step * 0.24);
      const accuracyValues = items.flatMap((item) => [item.oneStep?.type_accuracy || 0, item.rollout10?.type_accuracy || 0]);
      const rawMaxAccuracy = Math.max(
        0,
        ...accuracyValues
      );
      const rawMinAccuracy = Math.min(...accuracyValues);
      const accuracyRange = Math.max(0.02, rawMaxAccuracy - rawMinAccuracy);
      const minAccuracy = Math.max(0.0, rawMinAccuracy - accuracyRange * 0.18);
      const maxAccuracy = Math.min(1.0, Math.max(minAccuracy + 0.05, rawMaxAccuracy + accuracyRange * 0.12));
      const rawMaxTimeMae = Math.max(
        0,
        ...items.flatMap((item) => [item.oneStep?.time_mae || 0, item.rollout10?.time_mae || 0])
      );
      const maxTimeMae = Math.max(0.1, rawMaxTimeMae * 1.08);
      const accuracyTicks = [
        minAccuracy,
        minAccuracy + (maxAccuracy - minAccuracy) * 0.25,
        minAccuracy + (maxAccuracy - minAccuracy) * 0.5,
        minAccuracy + (maxAccuracy - minAccuracy) * 0.75,
        maxAccuracy,
      ].map((tick) => {{
        const y = accuracyTop + sectionHeight - sectionHeight * ((tick - minAccuracy) / Math.max(1e-9, maxAccuracy - minAccuracy));
        return `
          <line x1="${{margin.left}}" y1="${{y.toFixed(1)}}" x2="${{(chartWidth - margin.right).toFixed(1)}}" y2="${{y.toFixed(1)}}" stroke="rgba(31,41,51,0.08)" />
          <text x="${{(margin.left - 8).toFixed(1)}}" y="${{(y + 4).toFixed(1)}}" text-anchor="end" font-size="10" fill="#56616b">${{tick.toFixed(3)}}</text>
        `;
      }}).join("");
      const timeTicks = [0, maxTimeMae * 0.25, maxTimeMae * 0.5, maxTimeMae * 0.75, maxTimeMae].map((tick) => {{
        const y = timeTop + sectionHeight - sectionHeight * (tick / maxTimeMae);
        return `
          <line x1="${{margin.left}}" y1="${{y.toFixed(1)}}" x2="${{(chartWidth - margin.right).toFixed(1)}}" y2="${{y.toFixed(1)}}" stroke="rgba(31,41,51,0.08)" />
          <text x="${{(margin.left - 8).toFixed(1)}}" y="${{(y + 4).toFixed(1)}}" text-anchor="end" font-size="10" fill="#56616b">${{tick.toFixed(2)}}s</text>
        `;
      }}).join("");
      const bars = items.map((item, index) => {{
        const groupCenter = margin.left + step * index + step / 2;
        const oneStepAccuracy = item.oneStep?.type_accuracy || 0;
        const rolloutAccuracy = item.rollout10?.type_accuracy || 0;
        const oneStepMae = item.oneStep?.time_mae || 0;
        const rolloutMae = item.rollout10?.time_mae || 0;
        const oneStepAccuracyHeight = sectionHeight * ((oneStepAccuracy - minAccuracy) / Math.max(1e-9, maxAccuracy - minAccuracy));
        const rolloutAccuracyHeight = sectionHeight * ((rolloutAccuracy - minAccuracy) / Math.max(1e-9, maxAccuracy - minAccuracy));
        const oneStepMaeHeight = sectionHeight * (oneStepMae / maxTimeMae);
        const rolloutMaeHeight = sectionHeight * (rolloutMae / maxTimeMae);
        const labelY = timeTop + sectionHeight + 18;
        return `
          <rect x="${{(groupCenter - barWidth - 3).toFixed(1)}}" y="${{(accuracyTop + sectionHeight - oneStepAccuracyHeight).toFixed(1)}}"
                width="${{barWidth.toFixed(1)}}" height="${{oneStepAccuracyHeight.toFixed(1)}}" rx="4"
                fill="${{colors[item.name] || "#444"}}" fill-opacity="0.9" />
          <rect x="${{(groupCenter + 3).toFixed(1)}}" y="${{(accuracyTop + sectionHeight - rolloutAccuracyHeight).toFixed(1)}}"
                width="${{barWidth.toFixed(1)}}" height="${{rolloutAccuracyHeight.toFixed(1)}}" rx="4"
                fill="${{colors[item.name] || "#444"}}" fill-opacity="0.35" />
          <rect x="${{(groupCenter - barWidth - 3).toFixed(1)}}" y="${{(timeTop + sectionHeight - oneStepMaeHeight).toFixed(1)}}"
                width="${{barWidth.toFixed(1)}}" height="${{oneStepMaeHeight.toFixed(1)}}" rx="4"
                fill="${{colors[item.name] || "#444"}}" fill-opacity="0.9" />
          <rect x="${{(groupCenter + 3).toFixed(1)}}" y="${{(timeTop + sectionHeight - rolloutMaeHeight).toFixed(1)}}"
                width="${{barWidth.toFixed(1)}}" height="${{rolloutMaeHeight.toFixed(1)}}" rx="4"
                fill="${{colors[item.name] || "#444"}}" fill-opacity="0.35" />
          <text x="${{groupCenter.toFixed(1)}}" y="${{labelY.toFixed(1)}}" text-anchor="middle" font-size="10" fill="#56616b">${{shortNames[item.name] || item.name}}</text>
          <text x="${{groupCenter.toFixed(1)}}" y="${{(labelY + 12).toFixed(1)}}" text-anchor="middle" font-size="9" fill="#8b949e">${{item.oneStep?.predictions || 0}} obs</text>
        `;
      }}).join("");
      return `
        <svg viewBox="0 0 420 620" preserveAspectRatio="none">
          ${{accuracyTicks}}
          ${{timeTicks}}
          <text x="${{margin.left}}" y="${{(accuracyTop - 8).toFixed(1)}}" font-size="11" fill="#56616b">Type Accuracy</text>
          <text x="${{margin.left}}" y="${{(timeTop - 8).toFixed(1)}}" font-size="11" fill="#56616b">Time MAE</text>
          <line x1="${{margin.left}}" y1="${{(accuracyTop + sectionHeight).toFixed(1)}}" x2="${{(chartWidth - margin.right).toFixed(1)}}" y2="${{(accuracyTop + sectionHeight).toFixed(1)}}" stroke="rgba(31,41,51,0.16)" />
          <line x1="${{margin.left}}" y1="${{(timeTop + sectionHeight).toFixed(1)}}" x2="${{(chartWidth - margin.right).toFixed(1)}}" y2="${{(timeTop + sectionHeight).toFixed(1)}}" stroke="rgba(31,41,51,0.16)" />
          ${{bars}}
          <text x="${{chartWidth / 2}}" y="606" text-anchor="middle" font-size="10" fill="#56616b">solid = 1-step, faded = 10-event rollout</text>
        </svg>
      `;
    }}

    function renderFamilyCards() {{
      const familyNames = Array.from(
        new Set(
          modelNames.flatMap((name) => Object.keys(report.models[name].metrics.per_family || {{}}))
        )
      ).sort();
      familyGrid.innerHTML = "";
      familyNames.forEach((family) => {{
        const items = modelNames
          .map((name) => ({{
            name,
            oneStep: report.models[name].metrics.per_family?.[family],
            rollout10: report.models[name].long_horizon.rollout["10"].per_family?.[family],
          }}))
          .filter((item) => item.oneStep)
          .sort((a, b) => (b.oneStep?.type_accuracy || 0) - (a.oneStep?.type_accuracy || 0));
        const bestAccuracyOneStep = [...items].sort((a, b) => (b.oneStep?.type_accuracy || 0) - (a.oneStep?.type_accuracy || 0))[0];
        const bestAccuracyRollout = [...items].sort((a, b) => (b.rollout10?.type_accuracy || 0) - (a.rollout10?.type_accuracy || 0))[0];
        const bestMaeOneStep = [...items].sort((a, b) => (a.oneStep?.time_mae || 0) - (b.oneStep?.time_mae || 0))[0];
        const bestMaeRollout = [...items].sort((a, b) => (a.rollout10?.time_mae || 0) - (b.rollout10?.time_mae || 0))[0];
        const card = document.createElement("div");
        card.className = "family-card";
        card.innerHTML = `
          <h3>${{family}}</h3>
          <p>Best accuracy: <strong>${{bestAccuracyOneStep?.name || "—"}}</strong> one-step at ${{bestAccuracyOneStep?.oneStep?.type_accuracy?.toFixed(4) || "—"}}, <strong>${{bestAccuracyRollout?.name || "—"}}</strong> rollout-10 at ${{bestAccuracyRollout?.rollout10?.type_accuracy?.toFixed(4) || "—"}}. Lowest MAE: <strong>${{bestMaeOneStep?.name || "—"}}</strong> one-step at ${{bestMaeOneStep?.oneStep?.time_mae?.toFixed(4) || "—"}}s, <strong>${{bestMaeRollout?.name || "—"}}</strong> rollout-10 at ${{bestMaeRollout?.rollout10?.time_mae?.toFixed(4) || "—"}}s.</p>
          ${{buildFamilyChartSvg(items)}}
        `;
        familyGrid.appendChild(card);
      }});
    }}

    function buildConditionChartSvg(items) {{
      const chartHeight = 290;
      const chartWidth = 420;
      const margin = {{ top: 22, right: 18, bottom: 72, left: 42 }};
      const innerWidth = chartWidth - margin.left - margin.right;
      const innerHeight = chartHeight - margin.top - margin.bottom;
      const rawMax = Math.max(0, ...items.map((item) => item.balanced_accuracy || 0));
      const rawMin = Math.min(...items.map((item) => item.balanced_accuracy || 0));
      const valueRange = Math.max(0.02, rawMax - rawMin);
      const minValue = Math.max(0.0, rawMin - valueRange * 0.18);
      const maxValue = Math.min(1.0, Math.max(minValue + 0.05, rawMax + valueRange * 0.12));
      const step = innerWidth / Math.max(items.length, 1);
      const barWidth = Math.min(30, step * 0.52);
      const ticks = [
        minValue,
        minValue + (maxValue - minValue) * 0.25,
        minValue + (maxValue - minValue) * 0.5,
        minValue + (maxValue - minValue) * 0.75,
        maxValue,
      ].map((tick) => {{
        const y = margin.top + innerHeight - innerHeight * ((tick - minValue) / Math.max(1e-9, maxValue - minValue));
        return `
          <line x1="${{margin.left}}" y1="${{y.toFixed(1)}}" x2="${{(chartWidth - margin.right).toFixed(1)}}" y2="${{y.toFixed(1)}}" stroke="rgba(31,41,51,0.08)" />
          <text x="${{(margin.left - 8).toFixed(1)}}" y="${{(y + 4).toFixed(1)}}" text-anchor="end" font-size="10" fill="#56616b">${{tick.toFixed(3)}}</text>
        `;
      }}).join("");
      const bars = items.map((item, index) => {{
        const groupCenter = margin.left + step * index + step / 2;
        const height = innerHeight * (((item.balanced_accuracy || 0) - minValue) / Math.max(1e-9, maxValue - minValue));
        const y = margin.top + innerHeight - height;
        const labelY = margin.top + innerHeight + 18;
        return `
          <rect x="${{(groupCenter - barWidth / 2).toFixed(1)}}" y="${{y.toFixed(1)}}"
                width="${{barWidth.toFixed(1)}}" height="${{height.toFixed(1)}}" rx="5"
                fill="${{colors[item.name] || "#444"}}" fill-opacity="0.88" />
          <text x="${{groupCenter.toFixed(1)}}" y="${{labelY.toFixed(1)}}" text-anchor="middle" font-size="10" fill="#56616b">${{shortNames[item.name] || item.name}}</text>
          <text x="${{groupCenter.toFixed(1)}}" y="${{(labelY + 12).toFixed(1)}}" text-anchor="middle" font-size="9" fill="#8b949e">${{item.comparisons}} cmp</text>
        `;
      }}).join("");
      return `
        <svg viewBox="0 0 420 290" preserveAspectRatio="none">
          ${{ticks}}
          <text x="${{margin.left}}" y="${{(margin.top - 8).toFixed(1)}}" font-size="11" fill="#56616b">Balanced Accuracy</text>
          <line x1="${{margin.left}}" y1="${{(margin.top + innerHeight).toFixed(1)}}" x2="${{(chartWidth - margin.right).toFixed(1)}}" y2="${{(margin.top + innerHeight).toFixed(1)}}" stroke="rgba(31,41,51,0.16)" />
          ${{bars}}
        </svg>
      `;
    }}

    function renderConditionCards() {{
      const conditionNames = Object.keys(report.models[modelNames[0]].long_horizon.rollout["10"].conditions || {{}});
      const visibleModels = modelNames.filter((name) => activeConditionModels.has(name));
      const chartModels = visibleModels.length ? visibleModels : modelNames;
      conditionGrid.innerHTML = "";
      conditionNames.forEach((conditionName) => {{
        const items = chartModels
          .map((name) => ({{
            name,
            ...report.models[name].long_horizon.rollout["10"].conditions[conditionName],
          }}))
          .sort((a, b) => (b.balanced_accuracy || 0) - (a.balanced_accuracy || 0));
        const card = document.createElement("div");
        card.className = "condition-card";
        card.innerHTML = `
          <h3>${{conditionName.replaceAll("_", " ")}}</h3>
          <p>${{conditionDescriptions[conditionName] || "Condition-level forecast derived from the rolled-out future state."}}</p>
          ${{buildConditionChartSvg(items)}}
        `;
        conditionGrid.appendChild(card);
      }});
    }}

    function buildTimeConditionChartSvg(series) {{
      const chartHeight = 290;
      const chartWidth = 420;
      const margin = {{ top: 22, right: 18, bottom: 48, left: 42 }};
      const innerWidth = chartWidth - margin.left - margin.right;
      const innerHeight = chartHeight - margin.top - margin.bottom;
      const values = series.flatMap((item) => item.points.map((point) => point.value));
      const rawMax = Math.max(0, ...values);
      const rawMin = Math.min(...values);
      const valueRange = Math.max(0.02, rawMax - rawMin);
      const minValue = Math.max(0.0, rawMin - valueRange * 0.18);
      const maxValue = Math.min(1.0, Math.max(minValue + 0.05, rawMax + valueRange * 0.12));
      const xForIndex = (index) => margin.left + (index / 2) * innerWidth;
      const yForValue = (value) => margin.top + innerHeight - innerHeight * ((value - minValue) / Math.max(1e-9, maxValue - minValue));
      const ticks = [
        minValue,
        minValue + (maxValue - minValue) * 0.25,
        minValue + (maxValue - minValue) * 0.5,
        minValue + (maxValue - minValue) * 0.75,
        maxValue,
      ].map((tick) => {{
        const y = yForValue(tick);
        return `
          <line x1="${{margin.left}}" y1="${{y.toFixed(1)}}" x2="${{(chartWidth - margin.right).toFixed(1)}}" y2="${{y.toFixed(1)}}" stroke="rgba(31,41,51,0.08)" />
          <text x="${{(margin.left - 8).toFixed(1)}}" y="${{(y + 4).toFixed(1)}}" text-anchor="end" font-size="10" fill="#56616b">${{tick.toFixed(3)}}</text>
        `;
      }}).join("");
      const xLabels = ["10s", "30s", "60s"].map((label, index) => `
        <text x="${{xForIndex(index).toFixed(1)}}" y="${{(chartHeight - 18).toFixed(1)}}" text-anchor="middle" font-size="11" fill="#56616b">${{label}}</text>
      `).join("");
      const paths = series.map((item) => {{
        const points = item.points.map((point, index) => `${{xForIndex(index).toFixed(1)}},${{yForValue(point.value).toFixed(1)}}`).join(" ");
        const circles = item.points.map((point, index) => `
          <circle cx="${{xForIndex(index).toFixed(1)}}" cy="${{yForValue(point.value).toFixed(1)}}" r="4.5" fill="${{colors[item.name] || "#444"}}" />
        `).join("");
        return `
          <polyline points="${{points}}" fill="none" stroke="${{colors[item.name] || "#444"}}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />
          ${{circles}}
        `;
      }}).join("");
      return `
        <svg viewBox="0 0 420 290" preserveAspectRatio="none">
          ${{ticks}}
          <text x="${{margin.left}}" y="${{(margin.top - 8).toFixed(1)}}" font-size="11" fill="#56616b">Balanced Accuracy</text>
          <line x1="${{margin.left}}" y1="${{(margin.top + innerHeight).toFixed(1)}}" x2="${{(chartWidth - margin.right).toFixed(1)}}" y2="${{(margin.top + innerHeight).toFixed(1)}}" stroke="rgba(31,41,51,0.16)" />
          ${{paths}}
          ${{xLabels}}
        </svg>
      `;
    }}

    function renderTimeConditionCards() {{
      const horizonKeys = ["10s", "30s", "60s"];
      const firstModel = report.models[modelNames[0]];
      if (!firstModel.long_horizon.time_conditions) {{
        return;
      }}
      const conditionNames = Object.keys(firstModel.long_horizon.time_conditions["10s"] || {{}});
      const visibleModels = modelNames.filter((name) => activeConditionModels.has(name));
      const chartModels = visibleModels.length ? visibleModels : modelNames;
      timeConditionGrid.innerHTML = "";
      conditionNames.forEach((conditionName) => {{
        const series = chartModels.map((name) => ({{
          name,
          points: horizonKeys.map((horizon) => ({{
            horizon,
            value: report.models[name].long_horizon.time_conditions[horizon][conditionName].balanced_accuracy,
          }})),
        }}));
        const card = document.createElement("div");
        card.className = "condition-card";
        card.innerHTML = `
          <h3>${{conditionName.replaceAll("_", " ")}}</h3>
          <p>${{conditionDescriptions[conditionName] || "Condition-level forecast derived from the rolled-out future state."}}</p>
          ${{buildTimeConditionChartSvg(series)}}
        `;
        timeConditionGrid.appendChild(card);
      }});
    }}

    function buildProbabilisticConditionChartSvg(items) {{
      const width = 980;
      const height = 420;
      const margin = {{ top: 28, right: 24, bottom: 82, left: 54 }};
      const innerWidth = width - margin.left - margin.right;
      const sectionGap = 90;
      const sectionWidth = (innerWidth - sectionGap) / 2;
      const sectionHeight = height - margin.top - margin.bottom;
      const brierLeft = margin.left;
      const logLeft = margin.left + sectionWidth + sectionGap;
      const barStep = sectionWidth / Math.max(items.length, 1);
      const barWidth = Math.min(44, barStep * 0.55);
      const maxBrier = Math.max(0.05, Math.max(...items.map((item) => item.meanBrier)) * 1.12);
      const maxLogLoss = Math.max(0.1, Math.max(...items.map((item) => item.meanLogLoss)) * 1.12);
      const brierTicks = [0, maxBrier * 0.25, maxBrier * 0.5, maxBrier * 0.75, maxBrier].map((tick) => {{
        const y = margin.top + sectionHeight - sectionHeight * (tick / maxBrier);
        return `
          <line x1="${{brierLeft}}" y1="${{y.toFixed(1)}}" x2="${{(brierLeft + sectionWidth).toFixed(1)}}" y2="${{y.toFixed(1)}}" stroke="rgba(31,41,51,0.08)" />
          <text x="${{(brierLeft - 8).toFixed(1)}}" y="${{(y + 4).toFixed(1)}}" text-anchor="end" font-size="10" fill="#56616b">${{tick.toFixed(3)}}</text>
        `;
      }}).join("");
      const logTicks = [0, maxLogLoss * 0.25, maxLogLoss * 0.5, maxLogLoss * 0.75, maxLogLoss].map((tick) => {{
        const y = margin.top + sectionHeight - sectionHeight * (tick / maxLogLoss);
        return `
          <line x1="${{logLeft}}" y1="${{y.toFixed(1)}}" x2="${{(logLeft + sectionWidth).toFixed(1)}}" y2="${{y.toFixed(1)}}" stroke="rgba(31,41,51,0.08)" />
          <text x="${{(logLeft - 8).toFixed(1)}}" y="${{(y + 4).toFixed(1)}}" text-anchor="end" font-size="10" fill="#56616b">${{tick.toFixed(3)}}</text>
        `;
      }}).join("");
      const bars = items.map((item, index) => {{
        const brierX = brierLeft + barStep * index + (barStep - barWidth) / 2;
        const logX = logLeft + barStep * index + (barStep - barWidth) / 2;
        const brierHeight = sectionHeight * (item.meanBrier / maxBrier);
        const logHeight = sectionHeight * (item.meanLogLoss / maxLogLoss);
        const labelY = margin.top + sectionHeight + 18;
        return `
          <rect x="${{brierX.toFixed(1)}}" y="${{(margin.top + sectionHeight - brierHeight).toFixed(1)}}" width="${{barWidth.toFixed(1)}}" height="${{brierHeight.toFixed(1)}}" rx="5" fill="${{colors[item.name] || "#444"}}" fill-opacity="0.9" />
          <rect x="${{logX.toFixed(1)}}" y="${{(margin.top + sectionHeight - logHeight).toFixed(1)}}" width="${{barWidth.toFixed(1)}}" height="${{logHeight.toFixed(1)}}" rx="5" fill="${{colors[item.name] || "#444"}}" fill-opacity="0.9" />
          <text x="${{(brierX + barWidth / 2).toFixed(1)}}" y="${{labelY.toFixed(1)}}" text-anchor="middle" font-size="10" fill="#56616b">${{shortNames[item.name] || item.name}}</text>
          <text x="${{(logX + barWidth / 2).toFixed(1)}}" y="${{labelY.toFixed(1)}}" text-anchor="middle" font-size="10" fill="#56616b">${{shortNames[item.name] || item.name}}</text>
        `;
      }}).join("");
      return `
        <svg viewBox="0 0 980 420" preserveAspectRatio="none">
          ${{brierTicks}}
          ${{logTicks}}
          <text x="${{brierLeft}}" y="${{(margin.top - 8).toFixed(1)}}" font-size="12" fill="#56616b">Mean Brier</text>
          <text x="${{logLeft}}" y="${{(margin.top - 8).toFixed(1)}}" font-size="12" fill="#56616b">Mean Log Loss</text>
          <line x1="${{brierLeft}}" y1="${{(margin.top + sectionHeight).toFixed(1)}}" x2="${{(brierLeft + sectionWidth).toFixed(1)}}" y2="${{(margin.top + sectionHeight).toFixed(1)}}" stroke="rgba(31,41,51,0.16)" />
          <line x1="${{logLeft}}" y1="${{(margin.top + sectionHeight).toFixed(1)}}" x2="${{(logLeft + sectionWidth).toFixed(1)}}" y2="${{(margin.top + sectionHeight).toFixed(1)}}" stroke="rgba(31,41,51,0.16)" />
          ${{bars}}
        </svg>
      `;
    }}

    function renderProbabilisticConditionMetrics() {{
      const svg = document.getElementById("prob-condition-chart");
      const visibleModels = modelNames.filter((name) => activeConditionModels.has(name));
      const chartModels = visibleModels.length ? visibleModels : modelNames;
      const items = chartModels
        .map((name) => {{
          let brierSum = 0;
          let logLossSum = 0;
          let count = 0;
          Object.values(report.models[name].long_horizon.time_conditions || {{}}).forEach((horizonMetrics) => {{
            Object.values(horizonMetrics).forEach((stats) => {{
              brierSum += stats.brier || 0;
              logLossSum += stats.log_loss || 0;
              count += 1;
            }});
          }});
          return {{
            name,
            meanBrier: count ? brierSum / count : 0,
            meanLogLoss: count ? logLossSum / count : 0,
            count,
          }};
        }})
        .sort((a, b) => a.meanBrier - b.meanBrier);
      svg.outerHTML = buildProbabilisticConditionChartSvg(items);
      renderPerConditionProbabilisticCards(chartModels);
    }}

    function buildPerConditionProbCardSvg(items) {{
      const chartHeight = 620;
      const chartWidth = 420;
      const margin = {{ top: 28, right: 18, bottom: 78, left: 54 }};
      const innerWidth = chartWidth - margin.left - margin.right;
      const sectionGap = 64;
      const sectionHeight = (chartHeight - margin.top - margin.bottom - sectionGap) / 2;
      const brierTop = margin.top;
      const logTop = margin.top + sectionHeight + sectionGap;
      const step = innerWidth / Math.max(items.length, 1);
      const barWidth = Math.min(18, step * 0.24);
      const brierValues = items.map((item) => item.meanBrier || 0);
      const logValues = items.map((item) => item.meanLogLoss || 0);
      const maxBrier = Math.max(0.02, ...brierValues) * 1.08;
      const maxLogLoss = Math.max(0.05, ...logValues) * 1.08;
      const brierTicks = [0, maxBrier * 0.25, maxBrier * 0.5, maxBrier * 0.75, maxBrier].map((tick) => {{
        const y = brierTop + sectionHeight - sectionHeight * (tick / maxBrier);
        return `
          <line x1="${{margin.left}}" y1="${{y.toFixed(1)}}" x2="${{(chartWidth - margin.right).toFixed(1)}}" y2="${{y.toFixed(1)}}" stroke="rgba(31,41,51,0.08)" />
          <text x="${{(margin.left - 8).toFixed(1)}}" y="${{(y + 4).toFixed(1)}}" text-anchor="end" font-size="10" fill="#56616b">${{tick.toFixed(3)}}</text>
        `;
      }}).join("");
      const logTicks = [0, maxLogLoss * 0.25, maxLogLoss * 0.5, maxLogLoss * 0.75, maxLogLoss].map((tick) => {{
        const y = logTop + sectionHeight - sectionHeight * (tick / maxLogLoss);
        return `
          <line x1="${{margin.left}}" y1="${{y.toFixed(1)}}" x2="${{(chartWidth - margin.right).toFixed(1)}}" y2="${{y.toFixed(1)}}" stroke="rgba(31,41,51,0.08)" />
          <text x="${{(margin.left - 8).toFixed(1)}}" y="${{(y + 4).toFixed(1)}}" text-anchor="end" font-size="10" fill="#56616b">${{tick.toFixed(3)}}</text>
        `;
      }}).join("");
      const bars = items.map((item, index) => {{
        const centerX = margin.left + step * (index + 0.5);
        const x = centerX - barWidth / 2;
        const brierHeight = sectionHeight * ((item.meanBrier || 0) / maxBrier);
        const logHeight = sectionHeight * ((item.meanLogLoss || 0) / maxLogLoss);
        return `
          <rect x="${{x.toFixed(1)}}" y="${{(brierTop + sectionHeight - brierHeight).toFixed(1)}}" width="${{barWidth.toFixed(1)}}" height="${{brierHeight.toFixed(1)}}" rx="4" fill="${{colors[item.name] || "#444"}}" fill-opacity="0.9" />
          <rect x="${{x.toFixed(1)}}" y="${{(logTop + sectionHeight - logHeight).toFixed(1)}}" width="${{barWidth.toFixed(1)}}" height="${{logHeight.toFixed(1)}}" rx="4" fill="${{colors[item.name] || "#444"}}" fill-opacity="0.9" />
          <text x="${{centerX.toFixed(1)}}" y="${{(chartHeight - 18).toFixed(1)}}" text-anchor="middle" font-size="10" fill="#56616b">${{shortNames[item.name] || item.name}}</text>
        `;
      }}).join("");
      return `
        <svg viewBox="0 0 420 620" preserveAspectRatio="none">
          ${{brierTicks}}
          ${{logTicks}}
          <text x="${{margin.left}}" y="${{(brierTop - 8).toFixed(1)}}" font-size="11" fill="#56616b">Mean Brier</text>
          <text x="${{margin.left}}" y="${{(logTop - 8).toFixed(1)}}" font-size="11" fill="#56616b">Mean Log Loss</text>
          <line x1="${{margin.left}}" y1="${{(brierTop + sectionHeight).toFixed(1)}}" x2="${{(chartWidth - margin.right).toFixed(1)}}" y2="${{(brierTop + sectionHeight).toFixed(1)}}" stroke="rgba(31,41,51,0.16)" />
          <line x1="${{margin.left}}" y1="${{(logTop + sectionHeight).toFixed(1)}}" x2="${{(chartWidth - margin.right).toFixed(1)}}" y2="${{(logTop + sectionHeight).toFixed(1)}}" stroke="rgba(31,41,51,0.16)" />
          ${{bars}}
        </svg>
      `;
    }}

    function renderPerConditionProbabilisticCards(chartModels) {{
      const firstModel = report.models[modelNames[0]];
      const conditionNames = Object.keys(firstModel.long_horizon.time_conditions["10s"] || {{}});
      const horizonKeys = ["10s", "30s", "60s"];
      probConditionGrid.innerHTML = "";
      conditionNames.forEach((conditionName) => {{
        const items = chartModels.map((name) => {{
          let brierSum = 0;
          let logLossSum = 0;
          let count = 0;
          horizonKeys.forEach((horizon) => {{
            const stats = report.models[name].long_horizon.time_conditions[horizon][conditionName];
            brierSum += stats.brier || 0;
            logLossSum += stats.log_loss || 0;
            count += 1;
          }});
          return {{
            name,
            meanBrier: count ? brierSum / count : 0,
            meanLogLoss: count ? logLossSum / count : 0,
          }};
        }});
        const card = document.createElement("div");
        card.className = "condition-card";
        card.innerHTML = `
          <h3>${{conditionName.replaceAll("_", " ")}}</h3>
          <p>${{conditionDescriptions[conditionName] || "Condition-level forecast derived from the rolled-out future state."}}</p>
          ${{buildPerConditionProbCardSvg(items)}}
        `;
        probConditionGrid.appendChild(card);
      }});
    }}

    function buildCalibrationChartSvg(series) {{
      const width = 760;
      const height = 420;
      const margin = {{ top: 28, right: 20, bottom: 44, left: 50 }};
      const innerWidth = width - margin.left - margin.right;
      const innerHeight = height - margin.top - margin.bottom;
      const xFor = (value) => margin.left + innerWidth * value;
      const yFor = (value) => margin.top + innerHeight - innerHeight * value;
      const ticks = [0, 0.25, 0.5, 0.75, 1.0].map((tick) => `
        <line x1="${{xFor(0).toFixed(1)}}" y1="${{yFor(tick).toFixed(1)}}" x2="${{xFor(1).toFixed(1)}}" y2="${{yFor(tick).toFixed(1)}}" stroke="rgba(31,41,51,0.08)" />
        <line x1="${{xFor(tick).toFixed(1)}}" y1="${{yFor(0).toFixed(1)}}" x2="${{xFor(tick).toFixed(1)}}" y2="${{yFor(1).toFixed(1)}}" stroke="rgba(31,41,51,0.08)" />
        <text x="${{(margin.left - 8).toFixed(1)}}" y="${{(yFor(tick) + 4).toFixed(1)}}" text-anchor="end" font-size="10" fill="#56616b">${{tick.toFixed(2)}}</text>
        <text x="${{xFor(tick).toFixed(1)}}" y="${{(height - 18).toFixed(1)}}" text-anchor="middle" font-size="10" fill="#56616b">${{tick.toFixed(2)}}</text>
      `).join("");
      const diagonal = `<line x1="${{xFor(0).toFixed(1)}}" y1="${{yFor(0).toFixed(1)}}" x2="${{xFor(1).toFixed(1)}}" y2="${{yFor(1).toFixed(1)}}" stroke="#8b949e" stroke-dasharray="6 6" />`;
      const paths = series.map((item) => {{
        const validPoints = item.points.filter((point) => point.count > 0);
        const polyline = validPoints.map((point) => `${{xFor(point.mean_score).toFixed(1)}},${{yFor(point.actual_rate).toFixed(1)}}`).join(" ");
        const circles = validPoints.map((point) => `
          <circle cx="${{xFor(point.mean_score).toFixed(1)}}" cy="${{yFor(point.actual_rate).toFixed(1)}}" r="${{Math.max(3, Math.min(8, 2 + point.count / 20)).toFixed(1)}}" fill="${{colors[item.name] || '#444'}}" fill-opacity="0.85" />
        `).join("");
        return `
          <polyline points="${{polyline}}" fill="none" stroke="${{colors[item.name] || '#444'}}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />
          ${{circles}}
        `;
      }}).join("");
      return `
        <svg viewBox="0 0 760 420" preserveAspectRatio="none">
          ${{ticks}}
          ${{diagonal}}
          <text x="${{margin.left}}" y="${{(margin.top - 8).toFixed(1)}}" font-size="12" fill="#56616b">Observed Positive Rate</text>
          <text x="${{(width / 2).toFixed(1)}}" y="${{(height - 4).toFixed(1)}}" text-anchor="middle" font-size="12" fill="#56616b">Mean Predicted Probability</text>
          ${{paths}}
        </svg>
      `;
    }}

    function deriveCalibrationSummary(points) {{
      const valid = (points || []).filter((point) => (point.count || 0) > 0);
      const support = valid.reduce((sum, point) => sum + (point.count || 0), 0);
      if (!support) {{
        return {{ ece: 0, mce: 0, mean_confidence: 0, mean_observed_rate: 0, support: 0 }};
      }}
      let weightedGap = 0;
      let maxGap = 0;
      let confidenceSum = 0;
      let observedSum = 0;
      valid.forEach((point) => {{
        const gap = Math.abs((point.mean_score || 0) - (point.actual_rate || 0));
        weightedGap += gap * point.count;
        maxGap = Math.max(maxGap, gap);
        confidenceSum += (point.mean_score || 0) * point.count;
        observedSum += (point.actual_rate || 0) * point.count;
      }});
      return {{
        ece: weightedGap / support,
        mce: maxGap,
        mean_confidence: confidenceSum / support,
        mean_observed_rate: observedSum / support,
        support,
      }};
    }}

    function renderCalibrationSummaries() {{
      calibrationSummaryGrid.innerHTML = "";
      const items = modelNames.map((name) => {{
        const longHorizon = report.models[name].long_horizon || {{}};
        const summary = longHorizon.time_condition_calibration_summary || deriveCalibrationSummary(longHorizon.time_condition_calibration || []);
        return {{ name, summary }};
      }}).sort((a, b) => a.summary.ece - b.summary.ece);

      items.forEach((item) => {{
        const card = document.createElement("div");
        card.className = "condition-card";
        card.innerHTML = `
          <h3>${{shortNames[item.name] || item.name}}</h3>
          <p>Expected calibration error measures average probability mismatch. Max calibration error is the worst populated-bin mismatch.</p>
          <table>
            <tbody>
              <tr><td><strong>ECE</strong></td><td>${{item.summary.ece.toFixed(4)}}</td></tr>
              <tr><td><strong>Max CE</strong></td><td>${{item.summary.mce.toFixed(4)}}</td></tr>
              <tr><td><strong>Mean Conf.</strong></td><td>${{item.summary.mean_confidence.toFixed(4)}}</td></tr>
              <tr><td><strong>Obs. Rate</strong></td><td>${{item.summary.mean_observed_rate.toFixed(4)}}</td></tr>
              <tr><td><strong>Support</strong></td><td>${{item.summary.support}}</td></tr>
            </tbody>
          </table>
        `;
        calibrationSummaryGrid.appendChild(card);
      }});
    }}

    function renderCalibrationChart() {{
      const svg = document.getElementById("calibration-chart");
      const series = modelNames
        .filter((name) => report.models[name].long_horizon.time_condition_calibration)
        .map((name) => ({{
          name,
          points: report.models[name].long_horizon.time_condition_calibration,
      }}));
      svg.outerHTML = buildCalibrationChartSvg(series);
    }}

    function renderPerConditionCalibrationCards() {{
      calibrationGrid.innerHTML = "";
      Object.keys(conditionDescriptions).forEach((condition) => {{
        const series = modelNames
          .filter((name) => report.models[name].long_horizon.time_condition_calibration_by_condition?.[condition])
          .map((name) => ({{
            name,
            points: report.models[name].long_horizon.time_condition_calibration_by_condition[condition],
          }}));
        if (!series.length) {{
          return;
        }}
        const card = document.createElement("div");
        card.className = "condition-card";
        card.innerHTML = `
          <h3>${{condition}}</h3>
          <p>${{conditionDescriptions[condition] || "Per-condition reliability curve for calibrated probability forecasts."}}</p>
          ${{buildCalibrationChartSvg(series)}}
        `;
        calibrationGrid.appendChild(card);
      }});
    }}

    function renderTimelineChart() {{
      const svg = document.getElementById("timeline-chart");
      const width = 1100;
      const height = 520;
      const padding = {{ left: 90, right: 24, top: 28, bottom: 34 }};
      const rowGap = 88;
      const maxPoints = Math.min(70, report.actual_events.length);
      const actual = report.actual_events.slice(0, maxPoints);
      const rows = ["actual", ...modelNames];
      const maxTime = Math.max(...actual.map((item) => item.time_s), 1);

      svg.innerHTML = "";
      rows.forEach((row, rowIndex) => {{
        const y = padding.top + rowIndex * rowGap;
        svg.insertAdjacentHTML("beforeend", `<text x="10" y="${{y + 4}}" font-size="13" fill="#1f2933">${{row}}</text>`);
        svg.insertAdjacentHTML("beforeend", `<line x1="${{padding.left}}" y1="${{y}}" x2="${{width - padding.right}}" y2="${{y}}" stroke="rgba(31,41,51,0.08)" />`);
      }});

      actual.forEach((item) => {{
        const x = padding.left + (item.time_s / maxTime) * (width - padding.left - padding.right);
        const y = padding.top;
        svg.insertAdjacentHTML("beforeend", `<circle cx="${{x}}" cy="${{y}}" r="4" fill="#111" />`);
      }});

      modelNames.forEach((name, modelIndex) => {{
        const rowY = padding.top + (modelIndex + 1) * rowGap;
        report.models[name].example_predictions.slice(0, maxPoints).forEach((item) => {{
          const actualX = padding.left + (item.actual_time / maxTime) * (width - padding.left - padding.right);
          const predX = padding.left + (item.predicted_time / maxTime) * (width - padding.left - padding.right);
          svg.insertAdjacentHTML("beforeend", `<line x1="${{actualX}}" y1="${{rowY - 13}}" x2="${{predX}}" y2="${{rowY + 13}}" stroke="${{colors[name]}}" stroke-opacity="0.28" />`);
          svg.insertAdjacentHTML("beforeend", `<circle cx="${{actualX}}" cy="${{rowY - 13}}" r="3.2" fill="#111" />`);
          svg.insertAdjacentHTML("beforeend", `<circle cx="${{predX}}" cy="${{rowY + 13}}" r="4.1" fill="${{colors[name]}}" />`);
        }});
      }});
    }}

    setupConditionLegend("condition-legend");
    setupConditionLegend("time-condition-legend");
    setupConditionLegend("prob-condition-legend");
    syncConditionLegends();
    renderFamilyCards();
    renderConditionCards();
    renderTimeConditionCards();
    renderProbabilisticConditionMetrics();
    renderCalibrationSummaries();
    renderCalibrationChart();
    renderPerConditionCalibrationCards();
    renderTimelineChart();
  </script>
</body>
</html>
"""
