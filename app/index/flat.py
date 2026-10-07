from typing import Any, Dict, List, Optional
import numpy as np
from app.core.distance import batch_compute_distances, distance_to_score
from app.core.types import DistanceMetric, SearchResult, VectorRecord
from app.engine.filter import matches_filter
from app.index.base import VectorIndex

class FlatIndex(VectorIndex):
    """
    Exact, brute-force vector index with 100% recall.
    Uses contiguous NumPy arrays and BLAS vectorization for high speed.
    """

    def __init__(self, dimension: int, metric: DistanceMetric = DistanceMetric.COSINE):
        super().__init__(dimension, metric)
        # Vector matrix: shape (N, D), float32
        self.vectors: np.ndarray = np.empty((0, dimension), dtype=np.float32)
        # Maps id -> row index in self.vectors
        self.id_to_idx: Dict[str, int] = {}
        # Maps row index -> id
        self.idx_to_id: List[str] = []
        # Maps id -> metadata dict
        self.metadata_store: Dict[str, Dict[str, Any]] = {}

    def add(self, record_id: str, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        vec = np.asarray(vector, dtype=np.float32).flatten()
        if vec.shape[0] != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {vec.shape[0]}")

        meta = metadata if metadata is not None else {}

        if record_id in self.id_to_idx:
            # Update existing record
            idx = self.id_to_idx[record_id]
            self.vectors[idx] = vec
            self.metadata_store[record_id] = meta
        else:
            # Append new record
            idx = len(self.idx_to_id)
            if self.vectors.shape[0] == 0:
                self.vectors = np.expand_dims(vec, axis=0)
            else:
                self.vectors = np.vstack([self.vectors, vec])

            self.id_to_idx[record_id] = idx
            self.idx_to_id.append(record_id)
            self.metadata_store[record_id] = meta

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        if self.count() == 0 or k <= 0:
            return []

        q = np.asarray(query, dtype=np.float32).flatten()
        if q.shape[0] != self.dimension:
            raise ValueError(f"Query vector dimension mismatch: expected {self.dimension}, got {q.shape[0]}")

        # Compute distances for all vectors in one vectorized call
        distances = batch_compute_distances(q, self.vectors, self.metric)

        # Apply filtering if requested
        if filter_dict:
            valid_indices = [
                idx for idx, rec_id in enumerate(self.idx_to_id)
                if matches_filter(self.metadata_store.get(rec_id, {}), filter_dict)
            ]
            if not valid_indices:
                return []

            filtered_distances = distances[valid_indices]
            # Find top min(k, len(valid_indices))
            k_eff = min(k, len(valid_indices))
            top_local_idx = np.argsort(filtered_distances)[:k_eff]

            results = []
            for loc_idx in top_local_idx:
                actual_idx = valid_indices[loc_idx]
                rec_id = self.idx_to_id[actual_idx]
                dist = float(filtered_distances[loc_idx])
                results.append(
                    SearchResult(
                        id=rec_id,
                        score=distance_to_score(dist, self.metric),
                        distance=dist,
                        metadata=self.metadata_store.get(rec_id, {})
                    )
                )
            return results

        # No filter: fast top-k extraction
        k_eff = min(k, len(self.idx_to_id))
        top_indices = np.argsort(distances)[:k_eff]

        results = []
        for idx in top_indices:
            rec_id = self.idx_to_id[idx]
            dist = float(distances[idx])
            results.append(
                SearchResult(
                    id=rec_id,
                    score=distance_to_score(dist, self.metric),
                    distance=dist,
                    metadata=self.metadata_store.get(rec_id, {})
                )
            )
        return results

    def delete(self, record_id: str) -> bool:
        if record_id not in self.id_to_idx:
            return False

        idx = self.id_to_idx[record_id]
        last_idx = len(self.idx_to_id) - 1

        if idx != last_idx:
            # Swap with last element for O(1) delete
            last_id = self.idx_to_id[last_idx]
            self.vectors[idx] = self.vectors[last_idx]
            self.idx_to_id[idx] = last_id
            self.id_to_idx[last_id] = idx

        # Drop the last element
        self.vectors = self.vectors[:-1]
        self.idx_to_id.pop()
        del self.id_to_idx[record_id]
        if record_id in self.metadata_store:
            del self.metadata_store[record_id]

        return True

    def get(self, record_id: str) -> Optional[VectorRecord]:
        if record_id not in self.id_to_idx:
            return None
        idx = self.id_to_idx[record_id]
        return VectorRecord(
            id=record_id,
            vector=self.vectors[idx].tolist(),
            metadata=self.metadata_store.get(record_id, {})
        )

    def count(self) -> int:
        return len(self.idx_to_id)

    def clear(self) -> None:
        self.vectors = np.empty((0, self.dimension), dtype=np.float32)
        self.id_to_idx.clear()
        self.idx_to_id.clear()
        self.metadata_store.clear()

    def serialize(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "metric": self.metric.value,
            "vectors": self.vectors.tolist(),
            "idx_to_id": self.idx_to_id,
            "metadata_store": self.metadata_store
        }

    def deserialize(self, data: Dict[str, Any]) -> None:
        self.dimension = data["dimension"]
        self.metric = DistanceMetric(data["metric"])
        self.vectors = np.array(data["vectors"], dtype=np.float32) if data["vectors"] else np.empty((0, self.dimension), dtype=np.float32)
        self.idx_to_id = data["idx_to_id"]
        self.id_to_idx = {rec_id: idx for idx, rec_id in enumerate(self.idx_to_id)}
        self.metadata_store = data["metadata_store"]
