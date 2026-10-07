import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from app.core.types import DistanceMetric, IndexType, SearchResult, VectorRecord
from app.index.base import VectorIndex
from app.index.flat import FlatIndex
from app.index.hnsw import HNSWIndex
from app.storage.persistence import PersistenceManager
from app.storage.wal import WriteAheadLog

class Collection:
    """Thread-safe collection wrapping a vector index, WAL, and metadata persistence."""

    def __init__(
        self,
        name: str,
        dimension: int,
        metric: DistanceMetric,
        index_type: IndexType,
        data_dir: Path,
        auto_persist: bool = True
    ):
        self.name = name
        self.dimension = dimension
        self.metric = metric
        self.index_type = index_type
        self.data_dir = data_dir
        self.auto_persist = auto_persist
        self._lock = threading.RLock()

        # Initialize Index
        if index_type == IndexType.HNSW:
            self.index: VectorIndex = HNSWIndex(dimension=dimension, metric=metric)
        else:
            self.index = FlatIndex(dimension=dimension, metric=metric)

        # Storage components
        self.persistence = PersistenceManager(data_dir)
        self.wal = WriteAheadLog(data_dir / f"{name}.wal")

    def insert(self, record_id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            meta = metadata or {}
            # WAL write first for durability
            if self.auto_persist:
                self.wal.log_insert(record_id, vector, meta)

            # In-memory index update
            self.index.add(record_id, np.array(vector, dtype=np.float32), meta)

    def insert_batch(self, records: List[VectorRecord]) -> int:
        with self._lock:
            for r in records:
                self.insert(r.id, r.vector, r.metadata)
            return len(records)

    def search(
        self,
        query: List[float],
        k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        with self._lock:
            q_arr = np.array(query, dtype=np.float32)
            return self.index.search(q_arr, k=k, filter_dict=filter_dict)

    def delete(self, record_id: str) -> bool:
        with self._lock:
            success = self.index.delete(record_id)
            if success and self.auto_persist:
                self.wal.log_delete(record_id)
            return success

    def get(self, record_id: str) -> Optional[VectorRecord]:
        with self._lock:
            return self.index.get(record_id)

    def snapshot(self) -> None:
        """Trigger an on-disk snapshot and checkpoint the WAL."""
        with self._lock:
            self.persistence.save_snapshot(self.name, self.index)
            self.wal.clear()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            count = self.index.count()
            # Approximate memory footprint in bytes
            vector_bytes = count * self.dimension * 4
            return {
                "name": self.name,
                "count": count,
                "dimension": self.dimension,
                "metric": self.metric.value,
                "index_type": self.index_type.value,
                "estimated_memory_bytes": vector_bytes
            }


class CollectionManager:
    """Registry and lifecycle manager for all collections."""

    def __init__(self, data_dir: Path, auto_persist: bool = True):
        self.data_dir = data_dir
        self.auto_persist = auto_persist
        self._collections: Dict[str, Collection] = {}
        self._lock = threading.RLock()
        self.persistence = PersistenceManager(data_dir)
        self._load_existing_collections()

    def _load_existing_collections(self) -> None:
        """Scan data_dir for existing collection metadata and restore them."""
        for meta_file in self.data_dir.glob("*.meta.json"):
            name = meta_file.name.replace(".meta.json", "")
            meta = self.persistence.load_metadata(name)
            if meta:
                dim = meta["dimension"]
                metric = DistanceMetric(meta["metric"])
                idx_type = IndexType(meta.get("index_type", "flat"))

                coll = Collection(
                    name=name,
                    dimension=dim,
                    metric=metric,
                    index_type=idx_type,
                    data_dir=self.data_dir,
                    auto_persist=self.auto_persist
                )

                # Try loading snapshot
                has_snapshot = self.persistence.load_snapshot(name, coll.index)
                # Replay any WAL logs written after snapshot
                coll.wal.replay(coll.index)

                self._collections[name] = coll

    def create_collection(
        self,
        name: str,
        dimension: int,
        metric: DistanceMetric = DistanceMetric.COSINE,
        index_type: IndexType = IndexType.FLAT
    ) -> Collection:
        with self._lock:
            if name in self._collections:
                raise ValueError(f"Collection '{name}' already exists.")

            coll = Collection(
                name=name,
                dimension=dimension,
                metric=metric,
                index_type=index_type,
                data_dir=self.data_dir,
                auto_persist=self.auto_persist
            )

            if self.auto_persist:
                self.persistence.save_metadata(name, dimension, metric, index_type)

            self._collections[name] = coll
            return coll

    def get_collection(self, name: str) -> Optional[Collection]:
        with self._lock:
            return self._collections.get(name)

    def list_collections(self) -> List[str]:
        with self._lock:
            return list(self._collections.keys())

    def delete_collection(self, name: str) -> bool:
        with self._lock:
            if name not in self._collections:
                return False
            del self._collections[name]
            if self.auto_persist:
                self.persistence.delete_collection_files(name)
            return True
