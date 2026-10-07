# AegisVector DB 🛡️

A clean, high-performance, lightweight **Vector Database** written in Python using **FastAPI** and **NumPy**.

Featuring **HNSW (Hierarchical Navigable Small World)** logarithmic approximate search, **Flat** exact search, **MongoDB-style metadata filtering**, and **Write-Ahead Log (WAL) durability**.

---

## 🏗️ Architecture

```
vector-db/
├── app/
│   ├── api/
│   │   ├── routes.py          # FastAPI REST endpoints
│   │   └── schemas.py         # Pydantic validation schemas
│   ├── core/
│   │   ├── distance.py        # Vectorized Cosine, Euclidean & Dot Product (BLAS)
│   │   └── types.py           # Domain models & enums
│   ├── engine/
│   │   ├── collection.py      # Thread-safe collection management
│   │   └── filter.py          # Query filter evaluation ($eq, $gte, $in, $and, etc.)
│   ├── index/
│   │   ├── base.py            # VectorIndex abstract base class
│   │   ├── flat.py            # Exact linear scan index (100% recall)
│   │   └── hnsw.py            # HNSW multi-layer graph index (O(log N))
│   ├── storage/
│   │   ├── persistence.py     # Atomic snapshots & collection metadata
│   │   └── wal.py             # Append-only Write-Ahead Log for crash safety
│   ├── config.py              # Environment settings
│   └── main.py                # App entrypoint & visual dashboard
├── data/                      # On-disk WAL logs and snapshots
├── Dockerfile                 # Container image
├── docker-compose.yml         # Container orchestration with volume mount
├── requirements.txt           # Dependencies
└── test_db.py                 # Automated verification test suite
```

---

## ⚡ Quickstart

### 1. Run Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Start database server
uvicorn app.main:app --reload --port 8000
```

- **Interactive Dashboard**: Open [http://localhost:8000](http://localhost:8000)
- **Swagger OpenAPI Docs**: Open [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Run Tests
```bash
python test_db.py
```

### 3. Run with Docker
```bash
docker-compose up -d --build
```

---

## 📡 API Usage & Examples

### 1. Create a Collection
```bash
curl -X POST http://localhost:8000/api/v1/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "documents",
    "dimension": 4,
    "metric": "cosine",
    "index_type": "hnsw"
  }'
```
*Supported metrics*: `cosine`, `euclidean`, `dot`  
*Supported index types*: `hnsw` (graph ANN), `flat` (exact brute-force)

### 2. Insert Records with Metadata
```bash
curl -X POST http://localhost:8000/api/v1/collections/documents/insert \
  -H "Content-Type: application/json" \
  -d '{
    "id": "doc_101",
    "vector": [0.21, 0.54, -0.12, 0.88],
    "metadata": {
      "category": "science",
      "author": "Alice",
      "views": 1500
    }
  }'
```

### 3. Vector Similarity Query (with Metadata Filtering)
```bash
curl -X POST http://localhost:8000/api/v1/collections/documents/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.20, 0.50, -0.10, 0.85],
    "k": 5,
    "filter": {
      "category": "science",
      "views": { "$gte": 1000 }
    }
  }'
```

**Filter Operators Supported**:
- Comparison: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`
- Sets: `$in`, `$nin`
- Logical: `$and`, `$or`, `$not`

---

## 🐍 Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000/api/v1"

# 1. Create collection
requests.post(f"{BASE_URL}/collections", json={
    "name": "movies",
    "dimension": 3,
    "metric": "cosine",
    "index_type": "hnsw"
})

# 2. Insert vector
requests.post(f"{BASE_URL}/collections/movies/insert", json={
    "id": "matrix",
    "vector": [0.9, 0.1, 0.0],
    "metadata": {"genre": "Sci-Fi", "year": 1999}
})

# 3. Search
response = requests.post(f"{BASE_URL}/collections/movies/search", json={
    "vector": [0.85, 0.15, 0.05],
    "k": 1,
    "filter": {"genre": "Sci-Fi"}
})

print(response.json())
```

---

## 🚀 Deployment

Mount `./data` into `/app/data` to ensure all vector snapshots and write-ahead logs survive container restarts.

### DigitalOcean / AWS EC2 / VPS
```bash
git clone <your-repo>
cd vector-db
docker-compose up -d
```
All collection records and logs will be safely persisted in `./data` on your host machine.
