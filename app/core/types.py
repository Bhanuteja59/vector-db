from enum import Enum
from typing import Any, Dict, List
import numpy as np
from pydantic import BaseModel, Field

class DistanceMetric(str, Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot"

class IndexType(str, Enum):
    FLAT = "flat"
    HNSW = "hnsw"

class VectorRecord(BaseModel):
    id: str
    vector: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

class SearchResult(BaseModel):
    id: str
    score: float
    distance: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
