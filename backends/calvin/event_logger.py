"""Python-3.8-compatible structured event logger for CALVIN evaluation."""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path


class NullEventLogger:
    def emit(self, event, **fields):
        return None

    def close(self):
        return None


class JsonlEventLogger:
    def __init__(self, path, run_id):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)
        self._file = self.path.open("a")
        self._lock = threading.Lock()

    def emit(self, event, **fields):
        record = {
            "timestamp": datetime.now(timezone.utc)
            .astimezone()
            .isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "event": str(event),
        }
        record.update(fields)
        with self._lock:
            self._file.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            self._file.flush()

    def close(self):
        with self._lock:
            if not self._file.closed:
                self._file.close()
