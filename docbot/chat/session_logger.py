from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SessionLogger:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict[str, Any]) -> None:
        enriched = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(enriched, ensure_ascii=False) + "\n")
