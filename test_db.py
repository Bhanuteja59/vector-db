"""
Complete automated end-to-end verification test for AegisVector DB.
Tests: Flat Index, HNSW Index, Metadata Filtering, Persistence, and WAL recovery.
"""
import shutil
import tempfile
from pathlib import Path
import numpy as np

from app.core.types import DistanceMetric, IndexType
from app.engine.collection import CollectionManager

def run_tests():
    temp_dir = Path(tempfile.mkdtemp())
    print(f"[TEST] Testing AegisVector DB with temp directory: {temp_dir}")

    try:
        # 1. Initialize Collection Manager
        manager = CollectionManager(data_dir=temp_dir, auto_persist=True)

        # 2. Test Flat Collection
        print("\n--- 1. Testing Flat Index (Exact) ---")
        coll_flat = manager.create_collection(
            name="articles_flat",
            dimension=4,
            metric=DistanceMetric.COSINE,
            index_type=IndexType.FLAT
        )

        coll_flat.insert("doc_1", [1.0, 0.0, 0.0, 0.0], {"topic": "science", "year": 2024})
        coll_flat.insert("doc_2", [0.0, 1.0, 0.0, 0.0], {"topic": "history", "year": 2020})
        coll_flat.insert("doc_3", [0.9, 0.1, 0.0, 0.0], {"topic": "science", "year": 2023})
        coll_flat.insert("doc_4", [0.0, 0.0, 1.0, 0.0], {"topic": "art", "year": 2021})

        assert coll_flat.stats()["count"] == 4, "Count should be 4"
        print("[OK] Inserted 4 records into Flat collection.")

        # Search without filter: query close to [1.0, 0.0, 0.0, 0.0]
        results = coll_flat.search([1.0, 0.05, 0.0, 0.0], k=2)
        assert len(results) == 2
        assert results[0].id == "doc_1", f"Expected doc_1 as closest, got {results[0].id}"
        assert results[1].id == "doc_3", f"Expected doc_3 as second, got {results[1].id}"
        print(f"[OK] Flat search returned top 2: {[r.id for r in results]} (scores: {[round(r.score, 4) for r in results]})")

        # Search with metadata filter
        filtered_results = coll_flat.search(
            [1.0, 0.0, 0.0, 0.0],
            k=5,
            filter_dict={"topic": "science", "year": {"$gte": 2024}}
        )
        assert len(filtered_results) == 1
        assert filtered_results[0].id == "doc_1"
        print("[OK] Metadata filtering passed.")

        # 3. Test HNSW Graph Collection
        print("\n--- 2. Testing HNSW Graph Index (ANN) ---")
        coll_hnsw = manager.create_collection(
            name="articles_hnsw",
            dimension=8,
            metric=DistanceMetric.COSINE,
            index_type=IndexType.HNSW
        )

        np.random.seed(42)
        for i in range(50):
            vec = np.random.randn(8).tolist()
            coll_hnsw.insert(f"item_{i}", vec, {"idx": i, "group": "even" if i % 2 == 0 else "odd"})

        assert coll_hnsw.stats()["count"] == 50
        print("[OK] Inserted 50 high-dimensional vectors into HNSW graph.")

        # Query HNSW
        query_vec = np.random.randn(8).tolist()
        hnsw_results = coll_hnsw.search(query_vec, k=3)
        assert len(hnsw_results) == 3
        print(f"[OK] HNSW search returned top 3 candidates: {[r.id for r in hnsw_results]}")

        # Filtered HNSW search
        filtered_hnsw = coll_hnsw.search(query_vec, k=3, filter_dict={"group": "even"})
        assert all(r.metadata["group"] == "even" for r in filtered_hnsw)
        print("[OK] HNSW metadata filtered search successfully restricted results to 'even' group.")

        # 4. Test Record Deletion
        print("\n--- 3. Testing Deletion ---")
        assert coll_flat.delete("doc_2") is True
        assert coll_flat.stats()["count"] == 3
        assert coll_flat.get("doc_2") is None
        print("[OK] Deletion verified.")

        # 5. Test Persistence & Snapshot Recovery
        print("\n--- 4. Testing Snapshot & Crash Recovery ---")
        coll_flat.snapshot()
        coll_hnsw.snapshot()

        # Add one more item without snapshotting to test WAL replay
        coll_flat.insert("doc_wal", [0.5, 0.5, 0.0, 0.0], {"source": "wal_test"})

        # Re-instantiate CollectionManager from the same directory (simulating restart)
        reloaded_manager = CollectionManager(data_dir=temp_dir, auto_persist=True)
        reloaded_flat = reloaded_manager.get_collection("articles_flat")
        assert reloaded_flat is not None, "Failed to restore collection"
        assert reloaded_flat.stats()["count"] == 4, f"Expected 4 records after WAL replay, got {reloaded_flat.stats()['count']}"
        assert reloaded_flat.get("doc_wal") is not None, "WAL replayed record not found"
        print("[OK] Crash recovery & WAL replay successful! All data persisted across restarts.")

        print("\n[SUCCESS] ALL TESTS PASSED! AegisVector DB is fully operational.")

    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    run_tests()
