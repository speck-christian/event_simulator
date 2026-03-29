from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from .entities import EventRecord
from .viewer import build_viewer_html


def write_outputs(output_dir: Path, summary: dict, records: list[EventRecord]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "events.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["time_s", "event_type", "lane", "detail", "queue_after", "signal_phase"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    (output_dir / "index.html").write_text(build_viewer_html(summary, records))

