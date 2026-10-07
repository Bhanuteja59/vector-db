import json
from pathlib import Path
from typing import Any, Dict, Optional
from app.core.types import DistanceMetric, IndexType
from app.index.base import VectorIndex
from app.index.flat import FlatIndex
from app.index.hnsw import HNSWIndex

class PersistenceManager:
    """Manages collection metadata and disk snapshots."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _meta_path(self, collection_name: str) -> Path:
        return self.data_dir / f"{collection_name}.meta.json"

    def _snapshot_path(self, collection_name: str) -> Path:
        return self.data_dir / f"{collection_name}.snapshot.json"

    def save_metadata(
        self,
        name: str,
        dimension: int,
        metric: DistanceMetric,
        index_type: IndexType,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        meta = {
            "name": name,
            "dimension": dimension,
            "metric": metric.value,
            "index_type": index_type.value,
            **(extra or {})
        }
        with open(self._meta_path(name), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        path = self._meta_path(name)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_snapshot(self, name: str, index: VectorIndex) -> None:
        data = index.serialize()
        temp_path = self.data_dir / f"{name}.snapshot.tmp"
        final_path = self._snapshot_path(name)

        # Atomic write via rename
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        temp_path.replace(final_path)

    def load_snapshot(self, name: str, index: VectorIndex) -> bool:
        path = self._snapshot_path(name)
        if not path.exists():
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            index.deserialize(data)
        return True

    def delete_collection_files(self, name: str) -> None:
        for ext in [".meta.json", ".snapshot.json", ".snapshot.tmp", ".wal"]:
            p = self.data_dir / f"{name}{ext}"
            if p.exists():
                p.unlink()
