from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.core.types import DistanceMetric, IndexType, SearchResult

class CreateCollectionRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64, description="Unique collection name", pattern="^[a-zA-Z0-9_-]+$")
    dimension: int = Field(..., gt=0, le=10000, description="Vector dimension (e.g. 384, 768, 1536)")
    metric: DistanceMetric = Field(default=DistanceMetric.COSINE, description="Distance metric")
    index_type: IndexType = Field(default=IndexType.FLAT, description="Index type: 'flat' or 'hnsw'")

class CollectionStatsResponse(BaseModel):
    name: str
    dimension: int
    metric: DistanceMetric
    index_type: IndexType
    count: int
    estimated_memory_bytes: int

class InsertRecordRequest(BaseModel):
    id: str = Field(..., min_length=1, description="Unique record identifier")
    vector: List[float] = Field(..., description="Vector embedding float values")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom JSON metadata attributes")

class BatchInsertRequest(BaseModel):
    records: List[InsertRecordRequest] = Field(..., min_length=1, description="List of vector records to insert")

class SearchRequest(BaseModel):
    vector: List[float] = Field(..., description="Query vector embedding")
    k: int = Field(default=5, gt=0, le=500, description="Number of nearest neighbors to return")
    filter: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filter expression")

class SearchResponse(BaseModel):
    results: List[SearchResult]
    count: int
    latency_ms: float

class StatusResponse(BaseModel):
    success: bool
    message: str
