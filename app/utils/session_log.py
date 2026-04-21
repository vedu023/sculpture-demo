from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SessionLogger:
    def __init__(self, logs_dir: str | Path, command_name: str):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.logs_dir / f"session_{timestamp}_{command_name}.jsonl"
        self._handle = self.path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    def log_event(self, event_type: str, payload: dict[str, Any]):
        record = {
            "timestamp": datetime.now(tz=timezone.utc).astimezone().isoformat(),
            "event": event_type,
            **payload,
        }
        serialized = json.dumps(record, ensure_ascii=True, default=str)
        with self._lock:
            self._handle.write(serialized)
            self._handle.write("\n")
            self._handle.flush()

    def close(self):
        with self._lock:
            if not self._handle.closed:
                self._handle.flush()
                self._handle.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
