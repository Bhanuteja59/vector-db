from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
from app.core.types import DistanceMetric, SearchResult, VectorRecord

class VectorIndex(ABC):
    """Abstract base class defining interface for any vector index."""

    def __init__(self, dimension: int, metric: DistanceMetric):
        self.dimension = dimension
        self.metric = metric

    @abstractmethod
    def add(self, record_id: str, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add or update a vector record in the index."""
        pass

    @abstractmethod
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        keywords: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Search top-k nearest neighbors matching optional filter."""
        pass

    @abstractmethod
    def delete(self, record_id: str) -> bool:
        """Delete a vector record by id."""
        pass

    @abstractmethod
    def get(self, record_id: str) -> Optional[VectorRecord]:
        """Retrieve a vector record by id."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total number of vectors in index."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear the entire index."""
        pass

    @abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """Serialize index data for disk snapshot."""
        pass

    @abstractmethod
    def deserialize(self, data: Dict[str, Any]) -> None:
        """Restore index state from disk snapshot."""
        pass
