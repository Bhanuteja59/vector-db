"""
Integration test for FastAPI endpoints using TestClient.
Tests: Health, Dashboard HTML, API CRUD, Vector Search, and API Key Authentication.
"""
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings

def test_full_api():
    print("[TEST] Running FastAPI Live Integration Test...")

    with TestClient(app) as client:
        # 1. Health check
        res = client.get("/health")
        assert res.status_code == 200, f"Health check failed: {res.text}"
        data = res.json()
        assert data["status"] == "healthy"
        print("[OK] Health check endpoint passed (/health)")

        # 2. Visual Dashboard
        res = client.get("/")
        assert res.status_code == 200
        assert "AegisVector DB Dashboard" in res.text
        print("[OK] Web dashboard loaded successfully (/)")

        # 3. Create Collection
        coll_name = "test_movies"
        res = client.post("/api/v1/collections", json={
            "name": coll_name,
            "dimension": 3,
            "metric": "cosine",
            "index_type": "hnsw"
        })
        assert res.status_code == 200, f"Create collection failed: {res.text}"
        print(f"[OK] Created HNSW collection '{coll_name}'")

        # 4. Insert Batch
        res = client.post(f"/api/v1/collections/{coll_name}/insert/batch", json={
            "records": [
                {"id": "interstellar", "vector": [0.95, 0.1, 0.0], "metadata": {"genre": "Sci-Fi", "year": 2014}},
                {"id": "inception", "vector": [0.90, 0.15, 0.0], "metadata": {"genre": "Sci-Fi", "year": 2010}},
                {"id": "godfather", "vector": [0.0, 0.1, 0.95], "metadata": {"genre": "Crime", "year": 1972}}
            ]
        })
        assert res.status_code == 200, f"Batch insert failed: {res.text}"
        print("[OK] Inserted 3 movie vectors")

        # 5. Search Vector (Query for Sci-Fi near [0.92, 0.12, 0.0])
        res = client.post(f"/api/v1/collections/{coll_name}/search", json={
            "vector": [0.92, 0.12, 0.0],
            "k": 2,
            "filter": {"genre": "Sci-Fi"}
        })
        assert res.status_code == 200, f"Search failed: {res.text}"
        search_data = res.json()
        assert search_data["count"] == 2
        top_ids = [r["id"] for r in search_data["results"]]
        assert "interstellar" in top_ids and "inception" in top_ids
        print(f"[OK] Vector search successful: found {top_ids} (latency: {search_data['latency_ms']} ms)")

        # 6. Fetch Record by ID
        res = client.get(f"/api/v1/collections/{coll_name}/records/interstellar")
        assert res.status_code == 200
        assert res.json()["metadata"]["year"] == 2014
        print("[OK] Get record by ID passed")

        # 7. Delete Record by ID
        res = client.delete(f"/api/v1/collections/{coll_name}/records/godfather")
        assert res.status_code == 200
        print("[OK] Delete record by ID passed")

        # 8. Clean up collection
        res = client.delete(f"/api/v1/collections/{coll_name}")
        assert res.status_code == 200
        print(f"[OK] Cleaned up collection '{coll_name}'")

        # 9. Test API Key Security
        print("\n--- Testing API Key Security Enforcement ---")
        settings.API_KEY = "super-secret-key-123"
        try:
            # Request without key -> should be 401 Unauthorized
            res = client.get("/api/v1/collections")
            assert res.status_code == 401, f"Expected 401, got {res.status_code}"
            print("[OK] Request without API key blocked with HTTP 401 Unauthorized")

            # Request with wrong key -> should be 401
            res = client.get("/api/v1/collections", headers={"X-API-Key": "wrong-key"})
            assert res.status_code == 401
            print("[OK] Request with wrong API key blocked with HTTP 401 Unauthorized")

            # Request with valid key -> should be 200 OK
            res = client.get("/api/v1/collections", headers={"X-API-Key": "super-secret-key-123"})
            assert res.status_code == 200
            print("[OK] Request with valid 'X-API-Key' accepted with HTTP 200 OK")
        finally:
            settings.API_KEY = None  # Reset

    print("\n[SUCCESS] ALL FASTAPI ENDPOINTS & SECURITY CHECKS ARE WORKING 100% PERFECTLY!")

if __name__ == "__main__":
    test_full_api()
