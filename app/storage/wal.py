import json
from pathlib import Path
from typing import Any, Dict
import numpy as np
from app.index.base import VectorIndex

class WriteAheadLog:
    """
    Append-only log for write operations.
    Ensures crash safety by recording inserts and deletes before/after memory updates.
    """

    def __init__(self, wal_path: Path):
        self.wal_path = wal_path
        self.wal_path.parent.mkdir(parents=True, exist_ok=True)

    def log_insert(self, record_id: str, vector: list[float], metadata: Dict[str, Any]) -> None:
        entry = {
            "op": "insert",
            "id": record_id,
            "vector": vector,
            "metadata": metadata
        }
        self._append(entry)

    def log_batch_insert(self, records: list[Any]) -> None:
        lines = [
            json.dumps({"op": "insert", "id": r.id, "vector": r.vector, "metadata": r.metadata}) + "\n"
            for r in records
        ]
        with open(self.wal_path, "a", encoding="utf-8") as f:
            f.writelines(lines)
            f.flush()

    def log_delete(self, record_id: str) -> None:
        entry = {
            "op": "delete",
            "id": record_id
        }
        self._append(entry)

    def log_clear(self) -> None:
        entry = {"op": "clear"}
        self._append(entry)

    def _append(self, entry: Dict[str, Any]) -> None:
        with open(self.wal_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()

    def replay(self, index: VectorIndex) -> int:
        """Replay log entries to reconstruct index state. Returns number of replayed ops."""
        if not self.wal_path.exists():
            return 0

        replayed_count = 0
        with open(self.wal_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    op = entry.get("op")
                    if op == "insert":
                        index.add(
                            record_id=entry["id"],
                            vector=np.array(entry["vector"], dtype=np.float32),
                            metadata=entry.get("metadata", {})
                        )
                        replayed_count += 1
                    elif op == "delete":
                        index.delete(entry["id"])
                        replayed_count += 1
                    elif op == "clear":
                        index.clear()
                        replayed_count += 1
                except Exception:
                    # Ignore malformed lines from ungraceful abrupt termination
                    pass
        return replayed_count

    def clear(self) -> None:
        """Clear WAL file (used during checkpointing/snapshotting)."""
        if self.wal_path.exists():
            with open(self.wal_path, "w", encoding="utf-8") as f:
                f.truncate(0)
