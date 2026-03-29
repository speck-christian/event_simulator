from __future__ import annotations

import json
from dataclasses import asdict

from .entities import EventRecord


def build_viewer_html(summary: dict, records: list[EventRecord]) -> str:
    payload = {
        "summary": summary,
        "events": [asdict(record) for record in records],
    }
    data_json = json.dumps(payload).replace("</", "<\\/")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Traffic Intersection Viewer</title>
  <style>
    :root {{
      --bg: #efe8d8;
      --panel: rgba(255, 251, 244, 0.9);
      --road: #2c3136;
      --lane: #f5eed7;
      --grass: #99b97a;
      --accent: #b55331;
      --ink: #1f2933;
      --green: #2f9d59;
      --red: #d94b4b;
      --amber: #d6a12f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(255,255,255,0.5), transparent 30%),
        linear-gradient(135deg, #e7dcc2, var(--bg));
      min-height: 100vh;
    }}
    .page {{
      width: min(1200px, calc(100vw - 32px));
      margin: 24px auto;
      display: grid;
      grid-template-columns: 340px 1fr;
      gap: 18px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid rgba(31, 41, 51, 0.08);
      border-radius: 22px;
      box-shadow: 0 12px 34px rgba(74, 54, 32, 0.12);
      backdrop-filter: blur(8px);
    }}
    .sidebar {{
      padding: 22px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    h1 {{
      margin: 0;
      font-size: 1.8rem;
      line-height: 1;
      letter-spacing: 0.02em;
    }}
    .lede {{
      margin: 0;
      font-size: 0.98rem;
      line-height: 1.5;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }}
    button {{
      border: 0;
      border-radius: 999px;
      padding: 10px 16px;
      background: var(--accent);
      color: white;
      font: inherit;
      cursor: pointer;
    }}
    input[type="range"] {{
      width: 100%;
      accent-color: var(--accent);
    }}
    .stat-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }}
    .stat {{
      padding: 12px;
      border-radius: 16px;
      background: rgba(255,255,255,0.5);
    }}
    .stat strong {{
      display: block;
      font-size: 1.1rem;
    }}
    .lane-grid {{
      display: grid;
      gap: 10px;
    }}
    .lane-card {{
      border-radius: 16px;
      padding: 12px;
      background: rgba(255,255,255,0.55);
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: center;
    }}
    .lane-chip {{
      display: inline-block;
      min-width: 3.2rem;
      text-align: center;
      padding: 4px 8px;
      border-radius: 999px;
      color: white;
      font-size: 0.85rem;
      text-transform: capitalize;
    }}
    .viewer {{
      padding: 18px;
      position: relative;
      overflow: hidden;
    }}
    .scene {{
      position: relative;
      width: 100%;
      aspect-ratio: 1 / 1;
      border-radius: 28px;
      overflow: hidden;
      background:
        radial-gradient(circle at center, rgba(255,255,255,0.16), transparent 28%),
        linear-gradient(0deg, rgba(255,255,255,0.08), rgba(255,255,255,0.08)),
        var(--grass);
    }}
    .road.horizontal, .road.vertical {{
      position: absolute;
      background: var(--road);
    }}
    .road.horizontal {{ left: 0; right: 0; top: 36%; bottom: 36%; }}
    .road.vertical {{ top: 0; bottom: 0; left: 36%; right: 36%; }}
    .lane-mark {{
      position: absolute;
      background: repeating-linear-gradient(
        90deg,
        transparent 0 20px,
        rgba(247, 241, 219, 0.85) 20px 36px,
        transparent 36px 60px
      );
      opacity: 0.7;
    }}
    .lane-mark.h {{ left: 0; right: 0; top: calc(50% - 2px); height: 4px; }}
    .lane-mark.v {{
      top: 0;
      bottom: 0;
      left: calc(50% - 2px);
      width: 4px;
      background: repeating-linear-gradient(
        0deg,
        transparent 0 20px,
        rgba(247, 241, 219, 0.85) 20px 36px,
        transparent 36px 60px
      );
    }}
    .center-box {{
      position: absolute;
      left: 37%;
      top: 37%;
      width: 26%;
      height: 26%;
      border: 3px solid rgba(247, 241, 219, 0.9);
      border-radius: 18px;
    }}
    .stop-line {{
      position: absolute;
      background: rgba(255,255,255,0.92);
      border-radius: 999px;
    }}
    .stop-line.north, .stop-line.south {{
      width: 10%;
      height: 0.8%;
      left: 45%;
    }}
    .stop-line.north {{ top: 32%; }}
    .stop-line.south {{ top: 67.2%; }}
    .stop-line.east, .stop-line.west {{
      width: 0.8%;
      height: 10%;
      top: 45%;
    }}
    .stop-line.west {{ left: 32%; }}
    .stop-line.east {{ left: 67.2%; }}
    .signal-head {{
      position: absolute;
      width: 68px;
      padding: 8px;
      border-radius: 18px;
      background: rgba(28, 32, 36, 0.88);
      box-shadow: 0 10px 20px rgba(0,0,0,0.22);
    }}
    .signal-head.ns {{ top: 10%; right: 8%; }}
    .signal-head.ew {{ bottom: 10%; left: 8%; }}
    .signal-title {{
      color: white;
      font-size: 0.72rem;
      letter-spacing: 0.06em;
      margin-bottom: 6px;
      text-transform: uppercase;
    }}
    .bulbs {{
      display: grid;
      gap: 7px;
    }}
    .bulb {{
      width: 100%;
      aspect-ratio: 1 / 1;
      border-radius: 50%;
      background: rgba(255,255,255,0.12);
      border: 2px solid rgba(255,255,255,0.08);
    }}
    .bulb.red.active {{ background: var(--red); box-shadow: 0 0 16px rgba(217,75,75,0.65); }}
    .bulb.amber.active {{ background: var(--amber); box-shadow: 0 0 16px rgba(214,161,47,0.65); }}
    .bulb.green.active {{ background: var(--green); box-shadow: 0 0 16px rgba(47,157,89,0.65); }}
    .approach-label {{
      position: absolute;
      color: rgba(255,255,255,0.9);
      font-size: 0.78rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      background: rgba(31,41,51,0.34);
      padding: 6px 10px;
      border-radius: 999px;
    }}
    .approach-label.north {{ top: 5%; left: 50%; transform: translateX(-50%); }}
    .approach-label.south {{ bottom: 5%; left: 50%; transform: translateX(-50%); }}
    .approach-label.east {{ right: 4%; top: 50%; transform: translateY(-50%); }}
    .approach-label.west {{ left: 4%; top: 50%; transform: translateY(-50%); }}
    .layer {{
      position: absolute;
      inset: 0;
      pointer-events: none;
    }}
    .car {{
      position: absolute;
      width: 2.1%;
      height: 3.9%;
      border-radius: 8px;
      box-shadow: 0 6px 14px rgba(0,0,0,0.28);
      transform: translate(-50%, -50%);
      opacity: 0.96;
    }}
    .car.horizontal {{
      width: 3.9%;
      height: 2.1%;
    }}
    .car::after {{
      content: "";
      position: absolute;
      inset: 16%;
      border-radius: 5px;
      background: rgba(255,255,255,0.25);
    }}
    .timeline {{
      display: grid;
      gap: 8px;
    }}
    .timeline-row {{
      display: flex;
      justify-content: space-between;
      font-size: 0.92rem;
    }}
    .event-feed {{
      font-size: 0.95rem;
      line-height: 1.4;
      min-height: 3.2em;
      padding: 12px;
      border-radius: 14px;
      background: rgba(255,255,255,0.55);
    }}
    @media (max-width: 900px) {{
      .page {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <aside class="panel sidebar">
      <div>
        <h1>Intersection Replay</h1>
        <p class="lede">A browser animation for one simulated signal-control scenario. The replay uses the recorded event trace from this run.</p>
      </div>

      <div class="controls">
        <button id="toggle">Pause</button>
        <button id="restart">Restart</button>
      </div>

      <div>
        <label for="speed">Playback speed</label>
        <input id="speed" type="range" min="0.25" max="4" step="0.25" value="1.5" />
      </div>

      <div class="timeline">
        <input id="scrubber" type="range" min="0" max="1000" step="1" value="0" />
        <div class="timeline-row">
          <span id="clock">0.0 s</span>
          <span id="phase">NS_GREEN</span>
        </div>
      </div>

      <div class="stat-grid">
        <div class="stat"><span>Duration</span><strong id="duration"></strong></div>
        <div class="stat"><span>Events</span><strong id="events"></strong></div>
      </div>

      <div class="lane-grid" id="lane-grid"></div>

      <div class="event-feed" id="event-feed">Waiting for first event...</div>
    </aside>

    <main class="panel viewer">
      <div class="scene">
        <div class="road horizontal"></div>
        <div class="road vertical"></div>
        <div class="lane-mark h"></div>
        <div class="lane-mark v"></div>
        <div class="center-box"></div>
        <div class="stop-line north"></div>
        <div class="stop-line south"></div>
        <div class="stop-line east"></div>
        <div class="stop-line west"></div>

        <div class="approach-label north">Northbound Approach</div>
        <div class="approach-label south">Southbound Approach</div>
        <div class="approach-label east">Eastbound Approach</div>
        <div class="approach-label west">Westbound Approach</div>

        <div class="signal-head ns">
          <div class="signal-title">North / South</div>
          <div class="bulbs">
            <div class="bulb red" id="ns-red"></div>
            <div class="bulb amber" id="ns-amber"></div>
            <div class="bulb green" id="ns-green"></div>
          </div>
        </div>

        <div class="signal-head ew">
          <div class="signal-title">East / West</div>
          <div class="bulbs">
            <div class="bulb red" id="ew-red"></div>
            <div class="bulb amber" id="ew-amber"></div>
            <div class="bulb green" id="ew-green"></div>
          </div>
        </div>

        <div class="layer" id="queue-layer"></div>
        <div class="layer" id="motion-layer"></div>
      </div>
    </main>
  </div>

  <script id="simulation-data" type="application/json">{data_json}</script>
  <script>
    const raw = document.getElementById("simulation-data").textContent;
    const data = JSON.parse(raw);
    const events = data.events;
    const summary = data.summary;
    const laneColors = {{
      north: "#385f71",
      south: "#6b8e23",
      east: "#c75c38",
      west: "#7a4f9a",
    }};
    const lanes = ["north", "south", "east", "west"];
    const queueState = {{ north: 0, south: 0, east: 0, west: 0 }};
    const movingCars = [];
    let nextMovingCarId = 1;

    let currentTime = 0;
    let eventIndex = 0;
    let phase = "NS_GREEN";
    let playing = true;
    let speed = 1.5;
    let lastFrame = performance.now();
    const duration = summary.duration_seconds;

    const laneGrid = document.getElementById("lane-grid");
    lanes.forEach((lane) => {{
      const card = document.createElement("div");
      card.className = "lane-card";
      card.innerHTML = `
        <div>
          <span class="lane-chip" style="background:${{laneColors[lane]}}">${{lane}}</span>
          <div id="lane-text-${{lane}}">Queue 0</div>
        </div>
        <strong id="lane-wait-${{lane}}">${{summary.lanes[lane].average_wait_seconds}}s avg</strong>
      `;
      laneGrid.appendChild(card);
    }});

    document.getElementById("duration").textContent = `${{duration}} s`;
    document.getElementById("events").textContent = String(summary.events_recorded);

    function setActiveBulbs(group, activeColor) {{
      ["red", "amber", "green"].forEach((color) => {{
        document.getElementById(`${{group}}-${{color}}`).classList.toggle("active", color === activeColor);
      }});
    }}

    function laneOrientation(lane) {{
      return lane === "east" || lane === "west" ? "horizontal" : "vertical";
    }}

    function queuePosition(lane, index) {{
      const gap = 3.6;
      if (lane === "north") return {{ x: 48.3, y: 30 - index * gap }};
      if (lane === "south") return {{ x: 51.7, y: 70 + index * gap }};
      if (lane === "east") return {{ x: 70 + index * gap, y: 51.7 }};
      return {{ x: 30 - index * gap, y: 48.3 }};
    }}

    function travelPosition(lane, progress) {{
      const startEnd = {{
        north: {{ x1: 48.3, y1: 29, x2: 48.3, y2: 108 }},
        south: {{ x1: 51.7, y1: 71, x2: 51.7, y2: -8 }},
        east: {{ x1: 71, y1: 51.7, x2: -8, y2: 51.7 }},
        west: {{ x1: 29, y1: 48.3, x2: 108, y2: 48.3 }},
      }}[lane];
      return {{
        x: startEnd.x1 + (startEnd.x2 - startEnd.x1) * progress,
        y: startEnd.y1 + (startEnd.y2 - startEnd.y1) * progress,
      }};
    }}

    function makeCarElement(lane, x, y, scale = 1) {{
      const car = document.createElement("div");
      car.className = `car ${{laneOrientation(lane)}}`;
      car.style.left = `${{x}}%`;
      car.style.top = `${{y}}%`;
      car.style.background = laneColors[lane];
      car.style.transform = `translate(-50%, -50%) scale(${{scale}})`;
      return car;
    }}

    function renderQueues() {{
      const layer = document.getElementById("queue-layer");
      layer.innerHTML = "";
      lanes.forEach((lane) => {{
        const count = Math.min(queueState[lane], 20);
        for (let i = count - 1; i >= 0; i -= 1) {{
          const pos = queuePosition(lane, i);
          layer.appendChild(makeCarElement(lane, pos.x, pos.y));
        }}
        document.getElementById(`lane-text-${{lane}}`).textContent =
          `Queue ${{queueState[lane]}} | Max ${{summary.lanes[lane].max_queue}}`;
      }});
    }}

    function renderMovingCars() {{
      const layer = document.getElementById("motion-layer");
      layer.innerHTML = "";
      for (let i = movingCars.length - 1; i >= 0; i -= 1) {{
        const car = movingCars[i];
        const progress = (currentTime - car.start) / car.duration;
        if (progress >= 1) {{
          movingCars.splice(i, 1);
          continue;
        }}
        if (progress < 0) continue;
        const pos = travelPosition(car.lane, progress);
        layer.appendChild(makeCarElement(car.lane, pos.x, pos.y, 1.05));
      }}
    }}

    function renderSignals() {{
      if (phase === "NS_GREEN") {{
        setActiveBulbs("ns", "green");
        setActiveBulbs("ew", "red");
      }} else if (phase === "EW_GREEN") {{
        setActiveBulbs("ns", "red");
        setActiveBulbs("ew", "green");
      }} else {{
        setActiveBulbs("ns", "amber");
        setActiveBulbs("ew", "amber");
      }}
    }}

    function render() {{
      document.getElementById("clock").textContent = `${{currentTime.toFixed(1)}} s`;
      document.getElementById("phase").textContent = phase === "ALL_RED" ? "All directions red" : phase.replace("_", " ");
      document.getElementById("scrubber").value = Math.round((currentTime / duration) * 1000);
      renderSignals();
      renderQueues();
      renderMovingCars();
    }}

    function resetState() {{
      currentTime = 0;
      eventIndex = 0;
      phase = "NS_GREEN";
      lanes.forEach((lane) => {{
        queueState[lane] = 0;
      }});
      movingCars.length = 0;
      document.getElementById("event-feed").textContent = "Waiting for first event...";
      render();
    }}

    function applyEvent(event) {{
      if (event.event_type === "phase_change") {{
        phase = event.detail.split(";")[0].replace("phase=", "");
      }}
      if (event.lane && typeof event.queue_after === "number") {{
        queueState[event.lane] = event.queue_after;
      }}
      if (event.event_type === "vehicle_departure" && event.lane) {{
        movingCars.push({{
          id: nextMovingCarId++,
          lane: event.lane,
          start: event.time_s,
          duration: 3.2,
        }});
      }}
      const laneLabel = event.lane ? ` on ${{event.lane}}` : "";
      document.getElementById("event-feed").textContent =
        `${{event.event_type}}${{laneLabel}} at ${{event.time_s.toFixed(1)}}s. Signal state: ${{event.signal_phase}}.`;
    }}

    function seek(targetTime) {{
      resetState();
      currentTime = targetTime;
      while (eventIndex < events.length && events[eventIndex].time_s <= currentTime) {{
        applyEvent(events[eventIndex]);
        eventIndex += 1;
      }}
      render();
    }}

    function tick(now) {{
      const elapsed = (now - lastFrame) / 1000;
      lastFrame = now;
      if (playing) {{
        currentTime = Math.min(duration, currentTime + elapsed * speed * 6);
        while (eventIndex < events.length && events[eventIndex].time_s <= currentTime) {{
          applyEvent(events[eventIndex]);
          eventIndex += 1;
        }}
        render();
        if (currentTime >= duration) {{
          playing = false;
          document.getElementById("toggle").textContent = "Play";
        }}
      }}
      requestAnimationFrame(tick);
    }}

    document.getElementById("toggle").addEventListener("click", () => {{
      playing = !playing;
      document.getElementById("toggle").textContent = playing ? "Pause" : "Play";
    }});

    document.getElementById("restart").addEventListener("click", () => {{
      resetState();
      playing = true;
      document.getElementById("toggle").textContent = "Pause";
    }});

    document.getElementById("speed").addEventListener("input", (event) => {{
      speed = Number(event.target.value);
    }});

    document.getElementById("scrubber").addEventListener("input", (event) => {{
      const ratio = Number(event.target.value) / 1000;
      seek(duration * ratio);
    }});

    resetState();
    requestAnimationFrame((t) => {{
      lastFrame = t;
      requestAnimationFrame(tick);
    }});
  </script>
</body>
</html>
"""
