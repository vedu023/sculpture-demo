from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SessionLogger:
    def __init__(self, logs_dir: str | Path, command_name: str):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.logs_dir / f"session_{timestamp}_{command_name}.jsonl"

    def log_event(self, event_type: str, payload: dict[str, Any]):
        record = {
            "timestamp": datetime.now(tz=timezone.utc).astimezone().isoformat(),
            "event": event_type,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, default=str))
            handle.write("\n")
