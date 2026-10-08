import time
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, Depends
from app.api.schemas import (
    BatchInsertRequest,
    CollectionStatsResponse,
    CreateCollectionRequest,
    InsertRecordRequest,
    SearchRequest,
    SearchResponse,
    StatusResponse,
)
from app.core.types import DistanceMetric, IndexType, VectorRecord
from app.engine.collection import CollectionManager

from fastapi.security import APIKeyHeader
from app.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str | None = Depends(api_key_header)) -> None:
    """Enforce API key authentication if settings.API_KEY is configured."""
    if settings.API_KEY and settings.API_KEY.strip():
        if not api_key or api_key.strip() != settings.API_KEY.strip():
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: Invalid or missing 'X-API-Key' header."
            )

router = APIRouter(prefix="/api/v1", tags=["Vector DB"], dependencies=[Depends(verify_api_key)])

# Dependency placeholder (injected in main.py)
_manager: CollectionManager = None  # type: ignore

def set_manager(manager: CollectionManager) -> None:
    global _manager
    _manager = manager

def get_manager() -> CollectionManager:
    if _manager is None:
        raise RuntimeError("CollectionManager has not been initialized.")
    return _manager

# ----------------- Collections -----------------

@router.post("/collections", response_model=CollectionStatsResponse)
def create_collection(
    req: CreateCollectionRequest,
    manager: CollectionManager = Depends(get_manager)
):
    try:
        coll = manager.create_collection(
            name=req.name,
            dimension=req.dimension,
            metric=req.metric,
            index_type=req.index_type
        )
        return coll.stats()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/collections", response_model=List[str])
def list_collections(manager: CollectionManager = Depends(get_manager)):
    return manager.list_collections()

@router.get("/collections/{name}", response_model=CollectionStatsResponse)
def get_collection_stats(name: str, manager: CollectionManager = Depends(get_manager)):
    coll = manager.get_collection(name)
    if not coll:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found.")
    return coll.stats()

@router.delete("/collections/{name}", response_model=StatusResponse)
def delete_collection(name: str, manager: CollectionManager = Depends(get_manager)):
    success = manager.delete_collection(name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found.")
    return StatusResponse(success=True, message=f"Collection '{name}' deleted successfully.")

# ----------------- Vectors -----------------

@router.post("/collections/{name}/insert", response_model=StatusResponse)
def insert_record(
    name: str,
    req: InsertRecordRequest,
    manager: CollectionManager = Depends(get_manager)
):
    coll = manager.get_collection(name)
    if not coll:
        # Seamlessly auto-create collection if it doesn't exist
        try:
            coll = manager.create_collection(
                name=name,
                dimension=len(req.vector),
                metric=DistanceMetric(settings.DEFAULT_METRIC),
                index_type=IndexType(settings.DEFAULT_INDEX_TYPE)
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to auto-create collection '{name}': {str(e)}")

    if len(req.vector) != coll.dimension:
        raise HTTPException(
            status_code=400,
            detail=f"Dimension mismatch: collection expects {coll.dimension}, got {len(req.vector)}"
        )

    try:
        coll.insert(req.id, req.vector, req.metadata)
        return StatusResponse(success=True, message=f"Record '{req.id}' inserted into '{name}' successfully.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/collections/{name}/insert/batch", response_model=StatusResponse)
@router.post("/collections/{name}/batch_insert", response_model=StatusResponse)
def insert_batch(
    name: str,
    req: BatchInsertRequest,
    manager: CollectionManager = Depends(get_manager)
):
    if not req.records:
        raise HTTPException(status_code=400, detail="Batch records list cannot be empty.")

    coll = manager.get_collection(name)
    if not coll:
        # Seamlessly auto-create collection from first record's dimension
        try:
            coll = manager.create_collection(
                name=name,
                dimension=len(req.records[0].vector),
                metric=DistanceMetric(settings.DEFAULT_METRIC),
                index_type=IndexType(settings.DEFAULT_INDEX_TYPE)
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to auto-create collection '{name}': {str(e)}")

    records = []
    for r in req.records:
        if len(r.vector) != coll.dimension:
            raise HTTPException(
                status_code=400,
                detail=f"Record '{r.id}' dimension mismatch: expected {coll.dimension}, got {len(r.vector)}"
            )
        records.append(VectorRecord(id=r.id, vector=r.vector, metadata=r.metadata))

    inserted_count = coll.insert_batch(records)
    return StatusResponse(success=True, message=f"Successfully inserted {inserted_count} records into '{name}'.")

@router.post("/collections/{name}/search", response_model=SearchResponse)
def search_vectors(
    name: str,
    req: SearchRequest,
    manager: CollectionManager = Depends(get_manager)
):
    coll = manager.get_collection(name)
    if not coll:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found.")

    if len(req.vector) != coll.dimension:
        raise HTTPException(
            status_code=400,
            detail=f"Query vector dimension mismatch: collection expects {coll.dimension}, got {len(req.vector)}"
        )

    start_time = time.perf_counter()
    results = coll.search(req.vector, k=req.k, filter_dict=req.filter, keywords=req.keywords)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return SearchResponse(
        results=results,
        count=len(results),
        latency_ms=round(latency_ms, 3)
    )

@router.get("/collections/{name}/records/{record_id}", response_model=VectorRecord)
def get_record(
    name: str,
    record_id: str,
    manager: CollectionManager = Depends(get_manager)
):
    coll = manager.get_collection(name)
    if not coll:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found.")

    rec = coll.get(record_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Record '{record_id}' not found.")
    return rec

@router.delete("/collections/{name}/records/{record_id}", response_model=StatusResponse)
def delete_record(
    name: str,
    record_id: str,
    manager: CollectionManager = Depends(get_manager)
):
    coll = manager.get_collection(name)
    if not coll:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found.")

    success = coll.delete(record_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Record '{record_id}' not found.")
    return StatusResponse(success=True, message=f"Record '{record_id}' deleted successfully.")

@router.post("/collections/{name}/delete_by_filter", response_model=StatusResponse)
def delete_by_filter(
    name: str,
    filter_dict: Dict[str, Any],
    manager: CollectionManager = Depends(get_manager)
):
    coll = manager.get_collection(name)
    if not coll:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found.")

    count = coll.delete_by_filter(filter_dict)
    return StatusResponse(success=True, message=f"Deleted {count} record(s) matching filter from '{name}'.")

@router.post("/collections/{name}/snapshot", response_model=StatusResponse)
def snapshot_collection(name: str, manager: CollectionManager = Depends(get_manager)):
    coll = manager.get_collection(name)
    if not coll:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found.")

    coll.snapshot()
    return StatusResponse(success=True, message=f"Snapshot for collection '{name}' saved to disk.")
