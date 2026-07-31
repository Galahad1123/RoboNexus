"""Small structured event sinks used by task orchestrators."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _event_record(event: str, fields: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds"),
        "event": event,
        **fields,
    }


class NullEventLogger:
    def emit(self, event: str, **fields: Any) -> None:
        return None

    def close(self) -> None:
        return None


class InMemoryEventLogger:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event: str, **fields: Any) -> None:
        self.events.append(_event_record(event, fields))

    def close(self) -> None:
        return None


class JsonlEventLogger:
    """Append one complete JSON object per line and flush immediately."""

    def __init__(self, path: str | Path, run_id: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self._file = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def emit(self, event: str, **fields: Any) -> None:
        record = _event_record(event, {"run_id": self.run_id, **fields})
        with self._lock:
            self._file.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.close()
