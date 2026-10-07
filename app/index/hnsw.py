import heapq
import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from app.core.distance import compute_distance, distance_to_score
from app.core.types import DistanceMetric, SearchResult, VectorRecord
from app.engine.filter import matches_filter
from app.index.base import VectorIndex

class HNSWIndex(VectorIndex):
    """
    Hierarchical Navigable Small World (HNSW) graph index.
    Provides logarithmic O(log N) approximate nearest neighbor search.
    """

    def __init__(
        self,
        dimension: int,
        metric: DistanceMetric = DistanceMetric.COSINE,
        M: int = 16,
        ef_construction: int = 64,
        ef_search: int = 32
    ):
        super().__init__(dimension, metric)
        self.M = M
        self.M0 = 2 * M  # Max connections for layer 0
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.mL = 1.0 / math.log(M)

        # Graph storage: layer -> node_id -> set of neighbor node_ids
        # graphs[layer][id] = set(neighbor_ids)
        self.graphs: List[Dict[str, Set[str]]] = []
        self.entry_point: Optional[str] = None
        self.max_layer: int = -1

        # Node data
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata_store: Dict[str, Dict[str, Any]] = {}
        self.node_layers: Dict[str, int] = {}

    def _random_level(self) -> int:
        """Probabilistically choose the top layer for a newly inserted node."""
        r = random.random()
        if r == 0:
            r = 0.0000001
        return int(-math.log(r) * self.mL)

    def _dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return compute_distance(a, b, self.metric)

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: List[str],
        ef: int,
        layer: int
    ) -> List[Tuple[float, str]]:
        """
        Greedy beam search within a single graph layer.
        Returns list of (distance, node_id) sorted by distance.
        """
        visited: Set[str] = set(entry_points)
        # Min-heap of candidates: (dist, node_id)
        candidates: List[Tuple[float, str]] = []
        # Max-heap for dynamic top-ef results: (-dist, node_id)
        w: List[Tuple[float, str]] = []

        layer_graph = self.graphs[layer]

        for ep in entry_points:
            d = self._dist(query, self.vectors[ep])
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(w, (-d, ep))

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            # Furthest element in w
            worst_dist = -w[0][0]

            if c_dist > worst_dist:
                break

            for neighbor in layer_graph.get(c_id, ()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    worst_dist = -w[0][0]
                    d_neighbor = self._dist(query, self.vectors[neighbor])

                    if d_neighbor < worst_dist or len(w) < ef:
                        heapq.heappush(candidates, (d_neighbor, neighbor))
                        heapq.heappush(w, (-d_neighbor, neighbor))
                        if len(w) > ef:
                            heapq.heappop(w)

        # Sort results from best (smallest dist) to worst
        results = [(-neg_d, node_id) for neg_d, node_id in w]
        results.sort(key=lambda x: x[0])
        return results

    def add(self, record_id: str, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        vec = np.asarray(vector, dtype=np.float32).flatten()
        if vec.shape[0] != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {vec.shape[0]}")

        meta = metadata if metadata is not None else {}

        # If already exists, delete first to reinsert
        if record_id in self.vectors:
            self.delete(record_id)

        target_layer = self._random_level()
        self.vectors[record_id] = vec
        self.metadata_store[record_id] = meta
        self.node_layers[record_id] = target_layer

        # Ensure layers exist in self.graphs
        while len(self.graphs) <= max(target_layer, self.max_layer, 0):
            self.graphs.append({})

        # First node in index
        if self.entry_point is None:
            self.entry_point = record_id
            self.max_layer = target_layer
            for l in range(target_layer + 1):
                self.graphs[l][record_id] = set()
            return

        curr_obj = self.entry_point
        curr_dist = self._dist(vec, self.vectors[curr_obj])

        # Step 1: Zoom down from top layer to target_layer + 1 (greedy 1-best hop)
        for l in range(self.max_layer, target_layer, -1):
            changed = True
            while changed:
                changed = False
                for neighbor in self.graphs[l].get(curr_obj, ()):
                    d = self._dist(vec, self.vectors[neighbor])
                    if d < curr_dist:
                        curr_dist = d
                        curr_obj = neighbor
                        changed = True

        # Step 2: From min(target_layer, max_layer) down to layer 0, connect edges
        ep_list = [curr_obj]
        top_conn_layer = min(target_layer, self.max_layer)
        for l in range(top_conn_layer, -1, -1):
            m_max = self.M0 if l == 0 else self.M
            candidates = self._search_layer(vec, ep_list, self.ef_construction, l)
            ep_list = [c_id for _, c_id in candidates]

            # Choose up to m_max nearest neighbors
            neighbors = [c_id for _, c_id in candidates[:m_max]]
            self.graphs[l][record_id] = set(neighbors)

            # Bidirectional connection
            for n in neighbors:
                self.graphs[l].setdefault(n, set()).add(record_id)
                # Shrink if over max connections
                if len(self.graphs[l][n]) > m_max:
                    n_vec = self.vectors[n]
                    # Retain closest m_max
                    n_neighbors = list(self.graphs[l][n])
                    n_dists = [(self._dist(n_vec, self.vectors[n_cand]), n_cand) for n_cand in n_neighbors]
                    n_dists.sort(key=lambda x: x[0])
                    self.graphs[l][n] = set(cand for _, cand in n_dists[:m_max])

        # If new node has higher level than current max_layer, update entry point
        if target_layer > self.max_layer:
            for l in range(self.max_layer + 1, target_layer + 1):
                self.graphs[l][record_id] = set()
            self.entry_point = record_id
            self.max_layer = target_layer

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        if self.count() == 0 or k <= 0 or self.entry_point is None:
            return []

        q = np.asarray(query, dtype=np.float32).flatten()
        if q.shape[0] != self.dimension:
            raise ValueError(f"Query vector dimension mismatch: expected {self.dimension}, got {q.shape[0]}")

        curr_obj = self.entry_point
        curr_dist = self._dist(q, self.vectors[curr_obj])

        # Traverse down upper layers to layer 0
        for l in range(self.max_layer, 0, -1):
            changed = True
            while changed:
                changed = False
                for neighbor in self.graphs[l].get(curr_obj, ()):
                    d = self._dist(q, self.vectors[neighbor])
                    if d < curr_dist:
                        curr_dist = d
                        curr_obj = neighbor
                        changed = True

        # Search layer 0 with dynamic ef
        ef = max(self.ef_search, k * 2)
        if filter_dict:
            ef = max(ef, k * 5, 64)

        candidates = self._search_layer(q, [curr_obj], ef, 0)

        results: List[SearchResult] = []
        found_ids: Set[str] = set()

        for dist, rec_id in candidates:
            meta = self.metadata_store.get(rec_id, {})
            if filter_dict and not matches_filter(meta, filter_dict):
                continue

            results.append(
                SearchResult(
                    id=rec_id,
                    score=distance_to_score(dist, self.metric),
                    distance=dist,
                    metadata=meta
                )
            )
            found_ids.add(rec_id)
            if len(results) == k:
                break

        # If filtered results are fewer than k, check remaining matching items
        # to guarantee 100% recall even for highly selective metadata filters
        if filter_dict and len(results) < k:
            remaining: List[Tuple[float, str, Dict[str, Any]]] = []
            for rec_id, vec in self.vectors.items():
                if rec_id not in found_ids:
                    meta = self.metadata_store.get(rec_id, {})
                    if matches_filter(meta, filter_dict):
                        d = self._dist(q, vec)
                        remaining.append((d, rec_id, meta))

            if remaining:
                remaining.sort(key=lambda x: x[0])
                for dist, rec_id, meta in remaining:
                    results.append(
                        SearchResult(
                            id=rec_id,
                            score=distance_to_score(dist, self.metric),
                            distance=dist,
                            metadata=meta
                        )
                    )
                    if len(results) == k:
                        break

        return results

    def delete(self, record_id: str) -> bool:
        if record_id not in self.vectors:
            return False

        # Remove from graph layers
        node_layer = self.node_layers.get(record_id, 0)
        for l in range(min(node_layer + 1, len(self.graphs))):
            layer_graph = self.graphs[l]
            if record_id in layer_graph:
                neighbors = list(layer_graph[record_id])
                del layer_graph[record_id]
                for n in neighbors:
                    if n in layer_graph:
                        layer_graph[n].discard(record_id)

        del self.vectors[record_id]
        if record_id in self.metadata_store:
            del self.metadata_store[record_id]
        if record_id in self.node_layers:
            del self.node_layers[record_id]

        # Reset entry point if needed
        if self.entry_point == record_id:
            if not self.vectors:
                self.entry_point = None
                self.max_layer = -1
            else:
                # Pick any surviving node with highest layer
                best_id = max(self.node_layers.keys(), key=lambda x: self.node_layers[x])
                self.entry_point = best_id
                self.max_layer = self.node_layers[best_id]

        return True

    def get(self, record_id: str) -> Optional[VectorRecord]:
        if record_id not in self.vectors:
            return None
        return VectorRecord(
            id=record_id,
            vector=self.vectors[record_id].tolist(),
            metadata=self.metadata_store.get(record_id, {})
        )

    def count(self) -> int:
        return len(self.vectors)

    def clear(self) -> None:
        self.graphs.clear()
        self.vectors.clear()
        self.metadata_store.clear()
        self.node_layers.clear()
        self.entry_point = None
        self.max_layer = -1

    def serialize(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "metric": self.metric.value,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "entry_point": self.entry_point,
            "max_layer": self.max_layer,
            "node_layers": self.node_layers,
            "vectors": {k: v.tolist() for k, v in self.vectors.items()},
            "metadata_store": self.metadata_store,
            "graphs": [
                {node_id: list(neighbors) for node_id, neighbors in layer_g.items()}
                for layer_g in self.graphs
            ]
        }

    def deserialize(self, data: Dict[str, Any]) -> None:
        self.dimension = data["dimension"]
        self.metric = DistanceMetric(data["metric"])
        self.M = data.get("M", 16)
        self.M0 = 2 * self.M
        self.ef_construction = data.get("ef_construction", 64)
        self.ef_search = data.get("ef_search", 32)
        self.entry_point = data.get("entry_point")
        self.max_layer = data.get("max_layer", -1)
        self.node_layers = data.get("node_layers", {})
        self.vectors = {
            k: np.array(v, dtype=np.float32) for k, v in data.get("vectors", {}).items()
        }
        self.metadata_store = data.get("metadata_store", {})
        self.graphs = [
            {node_id: set(neighbors) for node_id, neighbors in layer_dict.items()}
            for layer_dict in data.get("graphs", [])
        ]
