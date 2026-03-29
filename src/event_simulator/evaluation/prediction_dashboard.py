from __future__ import annotations

import json
from typing import Any

from event_simulator.models.common import ReplayState
from event_simulator.models.common.conditions import CONDITION_NAMES, condition_flags


MODEL_COLORS = {
    "global_rate": "#8c5e34",
    "transition": "#2f6f85",
    "mechanistic": "#6d4c97",
    "neural_tpp": "#2d7f5e",
    "multitask_neural_tpp": "#d15a2e",
    "continuous_tpp": "#2466d1",
    "transformer_tpp": "#8f3bb8",
}

CONDITION_DESCRIPTIONS = {
    "congested": "Total queue reaches a congestion threshold across the whole intersection.",
    "severe_queue": "At least one lane reaches a visibly severe queue length.",
    "ns_pressure_high": "North-south demand remains meaningfully elevated.",
    "ew_pressure_high": "East-west demand remains meaningfully elevated.",
    "pressure_imbalance": "Directional pressure becomes materially unbalanced.",
}

HORIZONS = (10.0, 30.0, 60.0)


def actual_state_until_time(
    base_state: ReplayState,
    future_events: list[dict[str, Any]],
    summary: dict[str, Any],
    target_time: float,
) -> ReplayState:
    rolled_state = base_state.clone()
    for event in future_events:
        if float(event["time_s"]) > target_time:
            break
        rolled_state.update(event, summary)
    rolled_state.current_time = max(rolled_state.current_time, target_time)
    return rolled_state


def build_example_contexts(run: dict[str, Any]) -> list[dict[str, Any]]:
    state = ReplayState()
    contexts: list[dict[str, Any]] = []
    events = run["events"]
    duration = float(run["summary"]["duration_seconds"])
    for index, (current_event, next_event) in enumerate(zip(events, events[1:])):
        state.update(current_event, run["summary"])
        queue_state = {lane: int(value) for lane, value in state.queue_state.items()}
        future_events = events[index + 1 :]
        time_conditions: dict[str, dict[str, bool | None]] = {}
        for horizon in HORIZONS:
            target_time = state.current_time + horizon
            key = f"{int(horizon)}s"
            if target_time > duration:
                time_conditions[key] = {name: None for name in CONDITION_NAMES}
                continue
            target_state = actual_state_until_time(state, future_events, run["summary"], target_time)
            flags = condition_flags(target_state)
            time_conditions[key] = {name: bool(flags[name]) for name in CONDITION_NAMES}
        contexts.append(
            {
                "time_s": round(float(state.current_time), 3),
                "current_phase": state.current_phase,
                "phase_elapsed_s": round(max(0.0, state.current_time - state.phase_start_time), 3),
                "queue_state": queue_state,
                "total_queue": int(sum(queue_state.values())),
                "max_queue": int(max(queue_state.values())),
                "time_conditions": time_conditions,
                "next_departure_due": {
                    lane: (None if due is None else round(float(due), 3))
                    for lane, due in state.next_departure_due.items()
                },
                "next_actual_time": round(float(next_event["time_s"]), 3),
            }
        )
    return contexts


def build_prediction_dashboard_html(report: dict[str, Any], example_run: dict[str, Any], cached_runs: list[dict[str, Any]] | None = None) -> str:
    runs_for_dashboard = cached_runs or [example_run]
    run_payload = {}
    seed_order: list[int] = []
    for run in runs_for_dashboard:
        seed = int(run["seed"])
        seed_order.append(seed)
        run_payload[str(seed)] = {
            "seed": seed,
            "summary": run["summary"],
            "contexts": build_example_contexts(run),
        }
    payload = {
        "benchmark": {
            "train_runs": report["train_runs"],
            "eval_runs": report["eval_runs"],
            "duration_seconds": report["duration_seconds"],
            "example_seed": report["example_seed"],
            "control_mode": report.get("control_mode", "adaptive"),
            "simulation_profile": report.get("simulation_profile", "richer"),
        },
        "example_prediction_seed": int(report["example_seed"]),
        "seed_order": seed_order,
        "runs": run_payload,
        "models": report["models"],
        "model_colors": MODEL_COLORS,
        "condition_descriptions": CONDITION_DESCRIPTIONS,
        "condition_names": CONDITION_NAMES,
        "horizons": [f"{int(h)}s" for h in HORIZONS],
    }
    data_json = json.dumps(payload).replace("</", "<\\/")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Traffic Condition Forecast Dashboard</title>
  <style>
    :root {{
      --bg: #ece7df;
      --panel: rgba(255,255,255,0.86);
      --panel-strong: rgba(255,255,255,0.94);
      --ink: #18232d;
      --muted: #5b6872;
      --line: rgba(24,35,45,0.09);
      --road: #2a3038;
      --road-edge: #46515d;
      --signal: #f5f1e3;
      --accent: #ca6638;
      --good: #2e9b5e;
      --bad: #d1534b;
      --soft-good: rgba(46,155,94,0.15);
      --soft-bad: rgba(209,83,75,0.15);
      --gold: #d4a135;
      --north: #cf5a32;
      --south: #c6861c;
      --east: #2d7fa7;
      --west: #6f51b2;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top left, rgba(255,255,255,0.42), transparent 28%),
        linear-gradient(135deg, #ddd0bb, var(--bg));
    }}
    .page {{
      width: min(1440px, calc(100vw - 28px));
      margin: 18px auto 30px;
      display: grid;
      gap: 16px;
    }}
    .panel {{
      background: var(--panel);
      border-radius: 26px;
      border: 1px solid var(--line);
      box-shadow: 0 18px 44px rgba(67, 49, 30, 0.1);
      padding: 18px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.05fr 0.95fr;
      gap: 16px;
      align-items: start;
    }}
    h1, h2, h3 {{
      margin: 0 0 10px;
    }}
    .lede {{
      margin: 0;
      color: var(--muted);
      line-height: 1.58;
    }}
    .badge-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 12px;
    }}
    .badge {{
      background: rgba(255,255,255,0.66);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 0.92rem;
    }}
    .badge strong {{
      font-weight: 700;
    }}
    .control-grid {{
      display: grid;
      gap: 14px;
    }}
    .control-box {{
      background: rgba(255,255,255,0.62);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
    }}
    .row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .button-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    button {{
      appearance: none;
      border: 1px solid rgba(24,35,45,0.12);
      background: rgba(255,255,255,0.78);
      color: var(--ink);
      border-radius: 999px;
      padding: 8px 12px;
      font: inherit;
      cursor: pointer;
    }}
    select {{
      appearance: none;
      border: 1px solid rgba(24,35,45,0.12);
      background: rgba(255,255,255,0.9);
      color: var(--ink);
      border-radius: 999px;
      padding: 8px 12px;
      font: inherit;
    }}
    button.active {{
      color: white;
      border-color: transparent;
      box-shadow: 0 10px 20px rgba(0,0,0,0.14);
    }}
    input[type="range"] {{
      width: 100%;
      accent-color: var(--accent);
    }}
    .main-grid {{
      display: grid;
      grid-template-columns: 1.05fr 0.95fr;
      gap: 16px;
    }}
    .viz-stack {{
      display: grid;
      gap: 12px;
    }}
    .state-viz {{
      position: relative;
      width: 100%;
      aspect-ratio: 1 / 1;
      border-radius: 28px;
      overflow: hidden;
      border: 1px solid var(--line);
      background:
        radial-gradient(circle at center, rgba(255,255,255,0.14), transparent 26%),
        linear-gradient(145deg, #f4efe5 0%, #dce4d6 100%);
    }}
    .ring {{
      position: absolute;
      inset: 22%;
      border-radius: 50%;
      border: 18px solid rgba(255,255,255,0.28);
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.35);
    }}
    .road {{
      position: absolute;
      background: linear-gradient(180deg, var(--road-edge), var(--road));
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04);
    }}
    .road.ns {{ top: 0; bottom: 0; left: 40%; right: 40%; }}
    .road.ew {{ left: 0; right: 0; top: 40%; bottom: 40%; }}
    .center-core {{
      position: absolute;
      left: 35%;
      top: 35%;
      width: 30%;
      height: 30%;
      border-radius: 26px;
      background:
        radial-gradient(circle at 50% 50%, rgba(255,255,255,0.18), transparent 50%),
        linear-gradient(135deg, rgba(255,255,255,0.06), rgba(0,0,0,0.08));
      border: 2px solid rgba(245,241,227,0.72);
      box-shadow: 0 0 0 14px rgba(255,255,255,0.06);
    }}
    .phase-arc {{
      position: absolute;
      border-radius: 999px;
      opacity: 0.22;
      filter: blur(2px);
    }}
    .phase-arc.ns {{
      left: 45%;
      top: 6%;
      width: 10%;
      height: 88%;
      background: linear-gradient(180deg, rgba(46,155,94,0.9), rgba(46,155,94,0.15));
    }}
    .phase-arc.ew {{
      left: 6%;
      top: 45%;
      width: 88%;
      height: 10%;
      background: linear-gradient(90deg, rgba(46,155,94,0.9), rgba(46,155,94,0.15));
    }}
    .approach-band {{
      position: absolute;
      border-radius: 999px;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.22), 0 14px 28px rgba(0,0,0,0.12);
      overflow: hidden;
    }}
    .approach-band::after {{
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(90deg, rgba(255,255,255,0.14), rgba(255,255,255,0));
      pointer-events: none;
    }}
    .approach-band.north, .approach-band.south {{
      width: 12%;
      left: 44%;
    }}
    .approach-band.north {{ top: 8%; height: 24%; }}
    .approach-band.south {{ bottom: 8%; height: 24%; }}
    .approach-band.west, .approach-band.east {{
      height: 12%;
      top: 44%;
    }}
    .approach-band.west {{ left: 8%; width: 24%; }}
    .approach-band.east {{ right: 8%; width: 24%; }}
    .approach-fill {{
      position: absolute;
      inset: auto 0 0 0;
      border-radius: inherit;
    }}
    .approach-band.north .approach-fill,
    .approach-band.south .approach-fill {{
      inset: auto 0 0 0;
      width: 100%;
    }}
    .approach-band.west .approach-fill,
    .approach-band.east .approach-fill {{
      inset: 0 auto 0 0;
      height: 100%;
    }}
    .lane-label {{
      position: absolute;
      padding: 7px 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.84);
      border: 1px solid rgba(24,35,45,0.09);
      font-size: 0.82rem;
      letter-spacing: 0.03em;
      box-shadow: 0 12px 18px rgba(0,0,0,0.08);
    }}
    .lane-label.north {{ top: 3%; left: 50%; transform: translateX(-50%); }}
    .lane-label.south {{ bottom: 3%; left: 50%; transform: translateX(-50%); }}
    .lane-label.east {{ right: 3%; top: 50%; transform: translateY(-50%); }}
    .lane-label.west {{ left: 3%; top: 50%; transform: translateY(-50%); }}
    .mini-stats {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }}
    .mini-stat {{
      background: rgba(255,255,255,0.62);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px;
    }}
    .mini-stat strong {{
      display: block;
      font-size: 1.2rem;
    }}
    .congestion-stack {{
      display: grid;
      gap: 10px;
    }}
    .congestion-card {{
      background: rgba(255,255,255,0.68);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 12px 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}
    .congestion-card strong {{
      display: block;
      font-size: 1rem;
      margin-bottom: 2px;
    }}
    .congestion-card span {{
      color: var(--muted);
      font-size: 0.84rem;
      line-height: 1.35;
    }}
    .status-chip {{
      border-radius: 999px;
      padding: 7px 12px;
      font-size: 0.84rem;
      font-weight: 700;
      white-space: nowrap;
    }}
    .status-chip.on {{
      background: var(--soft-bad);
      color: #9b332f;
      border: 1px solid rgba(209,83,75,0.18);
    }}
    .status-chip.off {{
      background: var(--soft-good);
      color: #1f6d46;
      border: 1px solid rgba(46,155,94,0.18);
    }}
    .status-chip.na {{
      background: rgba(160,168,176,0.16);
      color: #5f6972;
      border: 1px solid rgba(160,168,176,0.2);
    }}
    .side-stack {{
      display: grid;
      gap: 12px;
      align-content: start;
    }}
    .horizon-card {{
      background: rgba(255,255,255,0.64);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
    }}
    .condition-pills {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 7px 10px;
      border-radius: 999px;
      font-size: 0.84rem;
      border: 1px solid transparent;
    }}
    .pill.on {{
      background: var(--soft-bad);
      color: #9b332f;
      border-color: rgba(209,83,75,0.18);
    }}
    .pill.off {{
      background: var(--soft-good);
      color: #1f6d46;
      border-color: rgba(46,155,94,0.18);
    }}
    .skill-card {{
      background: rgba(255,255,255,0.64);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
    }}
    .skill-header {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 10px;
    }}
    .skill-metric {{
      font-size: 0.86rem;
      color: var(--muted);
    }}
    .skill-grid {{
      display: grid;
      gap: 10px;
    }}
    .skill-row {{
      display: grid;
      grid-template-columns: 140px 1fr auto;
      gap: 10px;
      align-items: center;
    }}
    .bar-track {{
      position: relative;
      height: 10px;
      border-radius: 999px;
      background: rgba(24,35,45,0.08);
      overflow: hidden;
    }}
    .bar-fill {{
      position: absolute;
      inset: 0 auto 0 0;
      border-radius: inherit;
    }}
    .chart-card {{
      background: rgba(255,255,255,0.64);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 14px;
    }}
    svg {{
      width: 100%;
      height: auto;
      display: block;
      background: rgba(255,255,255,0.58);
      border-radius: 18px;
      border: 1px solid var(--line);
    }}
    .legend {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      font-size: 0.88rem;
      color: var(--muted);
      margin-top: 10px;
    }}
    .legend span::before {{
      content: "";
      width: 10px;
      height: 10px;
      display: inline-block;
      margin-right: 8px;
      border-radius: 50%;
      background: var(--color);
    }}
    @media (max-width: 1120px) {{
      .hero, .main-grid {{ grid-template-columns: 1fr; }}
      .mini-stats {{ grid-template-columns: 1fr; }}
      .skill-row {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="panel hero">
      <div>
        <h1>Traffic Condition Forecast Dashboard</h1>
        <p class="lede">This page focuses on future traffic regimes rather than next-event guesses. It shows what the intersection actually evolves into at each point in the example run, how difficult those future conditions are, and how well each model forecasts them at fixed horizons.</p>
        <div class="badge-row">
          <div class="badge">Seed <strong id="seed-badge"></strong></div>
          <div class="badge">Control <strong id="control-badge"></strong></div>
          <div class="badge">Profile <strong id="profile-badge"></strong></div>
          <div class="badge">Duration <strong id="duration-badge"></strong></div>
        </div>
      </div>
      <div class="control-grid">
        <div class="control-box">
          <h3>Example Run Cursor</h3>
          <div class="row" style="margin-bottom:10px;">
            <label for="seed-picker"><strong>Cached seed</strong></label>
            <select id="seed-picker"></select>
          </div>
          <div class="row">
            <span><strong id="step-label"></strong></span>
            <button id="play-toggle" type="button">Play</button>
          </div>
          <input id="step-slider" type="range" min="0" max="0" value="0" />
          <div class="row">
            <span id="context-time"></span>
            <span id="phase-name"></span>
          </div>
        </div>
        <div class="control-box">
          <h3>Forecast Focus</h3>
          <div class="button-row" id="model-chips"></div>
          <div class="button-row" id="horizon-chips" style="margin-top:10px;"></div>
        </div>
      </div>
    </section>

    <section class="main-grid">
      <div class="panel viz-stack">
        <div class="state-viz" id="state-viz"></div>
        <div class="mini-stats">
          <div class="mini-stat"><span>Total queue</span><strong id="total-queue"></strong></div>
          <div class="mini-stat"><span>Max lane queue</span><strong id="max-queue"></strong></div>
          <div class="mini-stat"><span>Phase elapsed</span><strong id="phase-elapsed"></strong></div>
        </div>
        <div class="congestion-stack" id="congestion-stack"></div>
      </div>

      <div class="panel side-stack">
        <div class="chart-card">
          <h2>Condition Timeline Comparison</h2>
          <p class="lede">Each condition is shown as a three-row block so you can compare actual now, actual future at the selected horizon, and the selected model's predicted future in one place.</p>
          <svg id="interleaved-condition-timeline" viewBox="0 0 980 620" preserveAspectRatio="none"></svg>
        </div>
        <div class="skill-card">
          <div class="skill-header">
            <div>
              <h3 id="selected-model-title"></h3>
              <p class="lede" id="selected-model-desc"></p>
            </div>
            <div class="skill-metric" id="selected-horizon-title"></div>
          </div>
          <div class="skill-grid" id="skill-grid"></div>
        </div>
      </div>
    </section>

    <section class="panel chart-card">
      <h2>Model Comparison For Selected Horizon</h2>
      <p class="lede">Balanced accuracy by condition for the selected future horizon. This compares condition skill directly across models on the same target definition.</p>
      <svg id="comparison-chart" viewBox="0 0 980 320" preserveAspectRatio="none"></svg>
    </section>
  </div>

  <script>
    const data = {data_json};
    const laneColors = {{
      north: getComputedStyle(document.documentElement).getPropertyValue("--north").trim(),
      south: getComputedStyle(document.documentElement).getPropertyValue("--south").trim(),
      east: getComputedStyle(document.documentElement).getPropertyValue("--east").trim(),
      west: getComputedStyle(document.documentElement).getPropertyValue("--west").trim(),
    }};
    const conditionNames = data.condition_names;
    const horizonNames = data.horizons;
    const modelOrder = Object.keys(data.models);
    const seedOrder = data.seed_order || [];
    let selectedSeed = String(data.benchmark.example_seed);
    let selectedModel = modelOrder.includes("multitask_neural_tpp") ? "multitask_neural_tpp" : modelOrder[0];
    let selectedHorizon = "30s";
    let selectedIndex = 0;
    let playTimer = null;

    function shortLabel(name) {{
      return {{
        global_rate: "Global",
        transition: "Transition",
        mechanistic: "Mechanistic",
        neural_tpp: "Neural",
        multitask_neural_tpp: "Multitask",
        continuous_tpp: "Continuous",
        transformer_tpp: "Transformer",
      }}[name] || name;
    }}

    function titleLabel(name) {{
      return shortLabel(name) + " Condition Forecasting";
    }}

    function currentRun() {{
      return data.runs[selectedSeed];
    }}

    function currentContext() {{
      return currentRun().contexts[selectedIndex];
    }}

    function initControls() {{
      document.getElementById("seed-badge").textContent = selectedSeed;
      document.getElementById("control-badge").textContent = data.benchmark.control_mode;
      document.getElementById("profile-badge").textContent = data.benchmark.simulation_profile;
      document.getElementById("duration-badge").textContent = `${{data.benchmark.duration_seconds}}s`;
      const seedPicker = document.getElementById("seed-picker");
      seedOrder.forEach((seed) => {{
        const option = document.createElement("option");
        option.value = String(seed);
        option.textContent = `Seed ${{seed}}`;
        seedPicker.appendChild(option);
      }});
      seedPicker.value = selectedSeed;
      seedPicker.addEventListener("change", (event) => {{
        selectedSeed = event.target.value;
        selectedIndex = 0;
        document.getElementById("step-slider").value = 0;
        renderAll();
      }});
      const slider = document.getElementById("step-slider");
      slider.max = Math.max(0, currentRun().contexts.length - 1);
      slider.addEventListener("input", (event) => {{
        selectedIndex = Number(event.target.value);
        renderAll();
      }});
      document.getElementById("play-toggle").addEventListener("click", togglePlay);
      const modelChips = document.getElementById("model-chips");
      modelOrder.forEach((name) => {{
        const button = document.createElement("button");
        button.textContent = shortLabel(name);
        button.dataset.modelName = name;
        button.addEventListener("click", () => {{
          selectedModel = name;
          renderAll();
        }});
        modelChips.appendChild(button);
      }});
      const horizonChips = document.getElementById("horizon-chips");
      horizonNames.forEach((name) => {{
        const button = document.createElement("button");
        button.textContent = name;
        button.dataset.horizon = name;
        button.addEventListener("click", () => {{
          selectedHorizon = name;
          renderAll();
        }});
        horizonChips.appendChild(button);
      }});
    }}

    function togglePlay() {{
      const button = document.getElementById("play-toggle");
      if (playTimer) {{
        window.clearInterval(playTimer);
        playTimer = null;
        button.textContent = "Play";
        return;
      }}
      playTimer = window.setInterval(() => {{
        selectedIndex = (selectedIndex + 1) % currentRun().contexts.length;
        document.getElementById("step-slider").value = selectedIndex;
        renderAll();
      }}, 600);
      button.textContent = "Pause";
    }}

    function renderControls() {{
      const run = currentRun();
      document.getElementById("seed-badge").textContent = selectedSeed;
      document.getElementById("seed-picker").value = selectedSeed;
      document.getElementById("step-slider").max = Math.max(0, run.contexts.length - 1);
      document.getElementById("step-slider").value = selectedIndex;
      document.getElementById("step-label").textContent = `Context ${{selectedIndex + 1}} / ${{run.contexts.length}}`;
      document.getElementById("context-time").textContent = `t = ${{currentContext().time_s.toFixed(1)}}s`;
      document.getElementById("phase-name").textContent = currentContext().current_phase;
      document.querySelectorAll("#model-chips button").forEach((button) => {{
        const active = button.dataset.modelName === selectedModel;
        button.classList.toggle("active", active);
        button.style.background = active ? (data.model_colors[selectedModel] || "#444") : "rgba(255,255,255,0.78)";
      }});
      document.querySelectorAll("#horizon-chips button").forEach((button) => {{
        const active = button.dataset.horizon === selectedHorizon;
        button.classList.toggle("active", active);
        if (active) {{
          button.style.background = "#ca6638";
          button.style.color = "white";
        }} else {{
          button.style.background = "rgba(255,255,255,0.78)";
          button.style.color = "var(--ink)";
        }}
      }});
    }}

    function renderStateViz() {{
      const context = currentContext();
      const viz = document.getElementById("state-viz");
      const queueNorm = (lane) => Math.min(1, context.queue_state[lane] / 18);
      const phaseNs = context.current_phase === "NS_GREEN";
      const phaseEw = context.current_phase === "EW_GREEN";
      viz.innerHTML = `
        <div class="road ns"></div>
        <div class="road ew"></div>
        <div class="ring"></div>
        <div class="center-core"></div>
        <div class="phase-arc ns" style="opacity:${{phaseNs ? 0.36 : 0.12}}"></div>
        <div class="phase-arc ew" style="opacity:${{phaseEw ? 0.36 : 0.12}}"></div>
        <div class="approach-band north">
          <div class="approach-fill" style="height:${{(queueNorm("north") * 100).toFixed(1)}}%; background:linear-gradient(180deg, rgba(207,90,50,0.35), rgba(207,90,50,0.95));"></div>
        </div>
        <div class="approach-band south">
          <div class="approach-fill" style="height:${{(queueNorm("south") * 100).toFixed(1)}}%; background:linear-gradient(180deg, rgba(198,134,28,0.35), rgba(198,134,28,0.95));"></div>
        </div>
        <div class="approach-band west">
          <div class="approach-fill" style="width:${{(queueNorm("west") * 100).toFixed(1)}}%; background:linear-gradient(90deg, rgba(111,81,178,0.35), rgba(111,81,178,0.95));"></div>
        </div>
        <div class="approach-band east">
          <div class="approach-fill" style="width:${{(queueNorm("east") * 100).toFixed(1)}}%; background:linear-gradient(90deg, rgba(45,127,167,0.35), rgba(45,127,167,0.95));"></div>
        </div>
        <div class="lane-label north">North · q${{context.queue_state.north}}</div>
        <div class="lane-label south">South · q${{context.queue_state.south}}</div>
        <div class="lane-label east">East · q${{context.queue_state.east}}</div>
        <div class="lane-label west">West · q${{context.queue_state.west}}</div>
      `;
      document.getElementById("total-queue").textContent = context.total_queue;
      document.getElementById("max-queue").textContent = context.max_queue;
      document.getElementById("phase-elapsed").textContent = `${{context.phase_elapsed_s.toFixed(1)}}s`;
    }}

    function formatStatusPill(value) {{
      if (value === null || value === undefined) {{
        return `<span class="status-chip na">n/a</span>`;
      }}
      return value
        ? `<span class="status-chip on">congested</span>`
        : `<span class="status-chip off">not congested</span>`;
    }}

    function renderCongestionStack() {{
      const context = currentContext();
      const hasPredictions = selectedSeed === String(data.example_prediction_seed);
      const predictions = hasPredictions ? (data.models[selectedModel].long_horizon.example_time_condition_predictions || []) : [];
      const predictedFlags = hasPredictions ? (predictions[selectedIndex]?.horizons?.[selectedHorizon]?.flags || null) : null;
      const currentCongested = context.total_queue >= 12;
      const futureCongested = context.time_conditions[selectedHorizon]?.congested ?? null;
      const predictedCongested = predictedFlags ? predictedFlags.congested : null;
      document.getElementById("congestion-stack").innerHTML = `
        <div class="congestion-card">
          <div>
            <strong>Actual Congestion Now</strong>
            <span>Uses the current queue state at this exact context. In this simulator, that means total queue \u2265 12 right now.</span>
          </div>
          ${{formatStatusPill(currentCongested)}}
        </div>
        <div class="congestion-card">
          <div>
            <strong>Actual Congestion At ${{selectedHorizon}}</strong>
            <span>This is the true future congestion label at the selected horizon, which can differ from the current queue if traffic is expected to drain or build.</span>
          </div>
          ${{formatStatusPill(futureCongested)}}
        </div>
        <div class="congestion-card">
          <div>
            <strong>${{shortLabel(selectedModel)}} Predicted Congestion At ${{selectedHorizon}}</strong>
            <span>${{hasPredictions ? "This comes from the selected model's own future condition forecast for the same horizon." : "Per-context model future-condition forecasts are only stored for the original example seed in this benchmark artifact."}}</span>
          </div>
          ${{formatStatusPill(predictedCongested)}}
        </div>
      `;
    }}

    function currentConditionFlags(context) {{
      const ns = context.queue_state.north + context.queue_state.south;
      const ew = context.queue_state.east + context.queue_state.west;
      const total = context.total_queue;
      const maxQueue = context.max_queue;
      return {{
        congested: total >= 12,
        severe_queue: maxQueue >= 6,
        ns_pressure_high: ns >= 8,
        ew_pressure_high: ew >= 8,
        pressure_imbalance: Math.abs(ns - ew) >= 5,
      }};
    }}

    function renderModelSkill() {{
      const model = data.models[selectedModel];
      const stats = model.long_horizon.time_conditions[selectedHorizon];
      document.getElementById("selected-model-title").textContent = titleLabel(selectedModel);
      document.getElementById("selected-model-desc").textContent = model.description;
      document.getElementById("selected-horizon-title").textContent = `Selected horizon: ${{selectedHorizon}}`;
      const grid = document.getElementById("skill-grid");
      grid.innerHTML = "";
      conditionNames.forEach((name) => {{
        const item = stats[name];
        const row = document.createElement("div");
        row.className = "skill-row";
        row.innerHTML = `
          <div>
            <strong>${{name}}</strong>
            <div style="color:var(--muted); font-size:0.82rem; margin-top:4px;">${{data.condition_descriptions[name]}}</div>
          </div>
          <div class="bar-track">
            <div class="bar-fill" style="width:${{(item.balanced_accuracy * 100).toFixed(1)}}%; background:${{data.model_colors[selectedModel] || "#555"}};"></div>
          </div>
          <div style="text-align:right;">
            <div><strong>${{item.balanced_accuracy.toFixed(3)}}</strong></div>
            <div style="font-size:0.82rem; color:var(--muted);">Brier ${{item.brier.toFixed(3)}}</div>
          </div>
        `;
        grid.appendChild(row);
      }});
    }}

    function renderInterleavedConditionTimeline() {{
      const svg = document.getElementById("interleaved-condition-timeline");
      const run = currentRun();
      const width = 980;
      const topPadding = 34;
      const bottomPadding = 30;
      const left = 220;
      const right = 18;
      const rowHeight = 28;
      const conditionGap = 18;
      const totalRows = conditionNames.length * 3;
      const height = topPadding + bottomPadding + totalRows * rowHeight + Math.max(0, conditionNames.length - 1) * conditionGap;
      const innerWidth = width - left - right;
      const cellWidth = innerWidth / Math.max(1, run.contexts.length);
      const cursorX = left + selectedIndex * cellWidth + cellWidth / 2;
      const hasPredictions = selectedSeed === String(data.example_prediction_seed);
      const predictions = hasPredictions ? (data.models[selectedModel].long_horizon.example_time_condition_predictions || []) : [];
      svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);

      function rowY(conditionIndex, variantIndex) {{
        return topPadding + conditionIndex * (3 * rowHeight + conditionGap) + variantIndex * rowHeight;
      }}

      const rowVariants = [
        {{
          label: "actual",
          getValue: (context, _index, name) => currentConditionFlags(context)[name],
          trueFill: "rgba(45,127,167,0.84)",
          falseFill: "rgba(45,127,167,0.24)",
        }},
        {{
          label: "actual future",
          getValue: (context, _index, name) => context.time_conditions[selectedHorizon][name],
          trueFill: "rgba(198,134,28,0.86)",
          falseFill: "rgba(198,134,28,0.24)",
        }},
        {{
          label: "predicted future",
          getValue: (_context, index, name) => {{
            const predictedFlags = hasPredictions ? (predictions[index]?.horizons?.[selectedHorizon]?.flags || null) : null;
            return predictedFlags ? predictedFlags[name] : null;
          }},
          trueFill: "rgba(111,81,178,0.86)",
          falseFill: "rgba(111,81,178,0.24)",
        }},
      ];

      const rows = conditionNames.map((name, conditionIndex) => {{
        const titleY = rowY(conditionIndex, 0) - 8;
        const sectionTitle = `<text x="18" y="${{titleY.toFixed(1)}}" font-size="12" font-weight="700" fill="#2a3840">${{name.replaceAll("_", " ")}}</text>`;
        const variantRows = rowVariants.map((variant, variantIndex) => {{
          const y = rowY(conditionIndex, variantIndex);
          const cells = run.contexts.map((context, index) => {{
            const value = variant.getValue(context, index, name);
            const fill = value === null
              ? "rgba(160,168,176,0.35)"
              : value
                ? variant.trueFill
                : variant.falseFill;
            return `<rect x="${{(left + index * cellWidth + 0.5).toFixed(1)}}" y="${{y.toFixed(1)}}" width="${{Math.max(1, cellWidth - 1).toFixed(1)}}" height="22" rx="4" fill="${{fill}}" opacity="${{index === selectedIndex ? 1 : 0.86}}" />`;
          }}).join("");
          return `
            <text x="${{left - 12}}" y="${{(y + 15).toFixed(1)}}" text-anchor="end" font-size="11" fill="#62707a">${{variant.label}}</text>
            ${{cells}}
          `;
        }}).join("");
        return sectionTitle + variantRows;
      }}).join("");

      svg.innerHTML = `
        <rect x="0" y="0" width="${{width}}" height="${{height}}" fill="rgba(255,255,255,0.6)" />
        <line x1="${{cursorX.toFixed(1)}}" y1="${{(topPadding - 10).toFixed(1)}}" x2="${{cursorX.toFixed(1)}}" y2="${{(height - bottomPadding + 4).toFixed(1)}}" stroke="rgba(24,35,45,0.42)" stroke-dasharray="5 5" />
        ${{rows}}
        <text x="${{left}}" y="18" font-size="12" fill="#62707a">Per condition: actual now, true future at ${{selectedHorizon}}, then ${{shortLabel(selectedModel)}} predicted future${{hasPredictions ? "" : " (n/a for this cached seed)"}}.</text>
        <text x="${{width - right}}" y="${{height - 10}}" text-anchor="end" font-size="11" fill="#62707a">example-run context index</text>
      `;
    }}

    function renderComparisonChart() {{
      const svg = document.getElementById("comparison-chart");
      const width = 980;
      const height = 320;
      const margin = {{ top: 28, right: 18, bottom: 44, left: 52 }};
      const innerWidth = width - margin.left - margin.right;
      const innerHeight = height - margin.top - margin.bottom;
      const groups = conditionNames;
      const barGroupWidth = innerWidth / groups.length;
      const activeModels = modelOrder.filter((name) => data.models[name].long_horizon?.time_conditions?.[selectedHorizon]);
      const barWidth = Math.min(16, (barGroupWidth - 12) / Math.max(1, activeModels.length));
      const values = activeModels.flatMap((name) => groups.map((condition) => data.models[name].long_horizon.time_conditions[selectedHorizon][condition].balanced_accuracy));
      const minVal = Math.max(0.45, Math.min(...values) - 0.03);
      const maxVal = Math.min(1.0, Math.max(...values) + 0.03);
      const y = (value) => margin.top + innerHeight - ((value - minVal) / Math.max(1e-6, maxVal - minVal)) * innerHeight;
      const grid = Array.from({{ length: 5 }}, (_, index) => {{
        const value = minVal + ((maxVal - minVal) / 4) * index;
        const yPos = y(value);
        return `<line x1="${{margin.left}}" y1="${{yPos.toFixed(1)}}" x2="${{width - margin.right}}" y2="${{yPos.toFixed(1)}}" stroke="rgba(24,35,45,0.08)" />
          <text x="${{margin.left - 8}}" y="${{(yPos + 4).toFixed(1)}}" text-anchor="end" font-size="11" fill="#62707a">${{value.toFixed(2)}}</text>`;
      }}).join("");
      const bars = groups.map((condition, groupIndex) => {{
        const groupLeft = margin.left + groupIndex * barGroupWidth;
        const pieces = activeModels.map((name, modelIndex) => {{
          const value = data.models[name].long_horizon.time_conditions[selectedHorizon][condition].balanced_accuracy;
          const barHeight = Math.max(2, margin.top + innerHeight - y(value));
          const x = groupLeft + 8 + modelIndex * barWidth;
          return `<rect x="${{x.toFixed(1)}}" y="${{y(value).toFixed(1)}}" width="${{Math.max(6, barWidth - 2).toFixed(1)}}" height="${{barHeight.toFixed(1)}}" rx="5" fill="${{data.model_colors[name] || "#666"}}" opacity="${{name === selectedModel ? 1 : 0.78}}" />`;
        }}).join("");
        return `
          ${{pieces}}
          <text x="${{(groupLeft + barGroupWidth / 2).toFixed(1)}}" y="${{height - 14}}" text-anchor="middle" font-size="11" fill="#62707a">${{condition.replace("_", " ")}}</text>
        `;
      }}).join("");
      const legend = activeModels.map((name, idx) => `<tspan x="${{margin.left + (idx % 3) * 190}}" dy="${{idx < 3 ? 0 : 16}}">● ${{shortLabel(name)}}</tspan>`).join("");
      svg.innerHTML = `
        <rect x="0" y="0" width="${{width}}" height="${{height}}" fill="rgba(255,255,255,0.6)" />
        ${{grid}}
        ${{bars}}
        <text x="${{margin.left}}" y="16" font-size="12" fill="#62707a">Balanced accuracy</text>
        <text x="${{margin.left}}" y="${{height - 70}}" font-size="11" fill="#62707a">${{selectedHorizon}} future condition forecast</text>
      `;
    }}

    function renderAll() {{
      renderControls();
      renderStateViz();
      renderCongestionStack();
      renderInterleavedConditionTimeline();
      renderModelSkill();
      renderComparisonChart();
    }}

    initControls();
    renderAll();
  </script>
</body>
</html>
"""
