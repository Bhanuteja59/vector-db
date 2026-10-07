from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.api.routes import router as api_router, set_manager
from app.config import settings
from app.engine.collection import CollectionManager

collection_manager: CollectionManager = None  # type: ignore

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize manager and recover collections from disk
    global collection_manager
    collection_manager = CollectionManager(
        data_dir=settings.DATA_DIR,
        auto_persist=settings.AUTO_PERSIST
    )
    set_manager(collection_manager)
    print(f"[INFO] {settings.APP_NAME} v{settings.VERSION} initialized.")
    print(f"[INFO] Storage directory: {settings.DATA_DIR.resolve()}")
    print(f"[INFO] Loaded {len(collection_manager.list_collections())} collection(s).")
    yield
    # Shutdown: Trigger snapshots for all collections
    print("[INFO] Graceful shutdown: saving snapshots to disk...")
    for name in collection_manager.list_collections():
        coll = collection_manager.get_collection(name)
        if coll:
            coll.snapshot()
    print("[INFO] Snapshots saved. Server shutdown.")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="High-performance Vector Database with HNSW, Flat index, and Metadata Filtering built on FastAPI & NumPy.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.VERSION,
        "collections": len(collection_manager.list_collections()) if collection_manager else 0
    }

@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
def dashboard():
    """Interactive visual Web Dashboard for exploring the Vector DB."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AegisVector DB Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #090d16;
                --surface: #111827;
                --border: #1f2937;
                --accent: #6366f1;
                --accent-hover: #4f46e5;
                --success: #10b981;
                --text: #f3f4f6;
                --text-muted: #9ca3af;
            }
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
            body { background: var(--bg); color: var(--text); padding: 2rem; }
            .container { max-width: 1200px; margin: 0 auto; }
            header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; padding-bottom: 1.5rem; border-bottom: 1px solid var(--border); }
            h1 { font-size: 1.75rem; font-weight: 700; background: linear-gradient(135deg, #a5b4fc, #6366f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            .badge { background: #1e1b4b; color: #a5b4fc; border: 1px solid #3730a3; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.85rem; font-weight: 600; }
            .links a { color: #818cf8; text-decoration: none; margin-left: 1rem; font-size: 0.95rem; font-weight: 500; }
            .links a:hover { text-decoration: underline; }
            .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }
            .card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; }
            .card h2 { font-size: 1.15rem; margin-bottom: 1rem; color: #e5e7eb; display: flex; align-items: center; justify-content: space-between; }
            .form-group { margin-bottom: 1rem; }
            label { display: block; font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.4rem; font-weight: 500; }
            input, select, textarea { width: 100%; background: #0b0f19; border: 1px solid var(--border); border-radius: 8px; padding: 0.65rem 0.85rem; color: var(--text); font-size: 0.9rem; }
            input:focus, select:focus, textarea:focus { border-color: var(--accent); outline: none; }
            button { background: var(--accent); color: white; border: none; padding: 0.65rem 1.25rem; border-radius: 8px; font-weight: 600; cursor: pointer; transition: background 0.15s; font-size: 0.9rem; }
            button:hover { background: var(--accent-hover); }
            .collection-item { background: #0b0f19; border: 1px solid var(--border); padding: 0.9rem 1.2rem; border-radius: 8px; margin-bottom: 0.75rem; display: flex; justify-content: space-between; align-items: center; }
            .collection-info { display: flex; flex-direction: column; gap: 0.2rem; }
            .collection-name { font-weight: 600; font-size: 1rem; color: #fff; }
            .collection-meta { font-size: 0.8rem; color: var(--text-muted); }
            .results-box { background: #0b0f19; border: 1px solid var(--border); border-radius: 8px; padding: 1rem; max-height: 280px; overflow-y: auto; font-family: monospace; font-size: 0.85rem; white-space: pre-wrap; color: #a5b4fc; }
            .status-msg { margin-top: 0.5rem; font-size: 0.85rem; }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <div>
                    <h1>AegisVector DB</h1>
                    <p style="color: var(--text-muted); font-size: 0.9rem; margin-top: 0.2rem;">Native Fast Vector Database Engine</p>
                </div>
                <div class="links" style="display: flex; align-items: center; gap: 0.75rem;">
                    <input id="api-key-input" placeholder="API Key (optional)" style="width: 170px; padding: 0.35rem 0.65rem; font-size: 0.8rem;" onchange="saveApiKey()" />
                    <span class="badge" style="background: #064e3b; color: #6ee7b7; border-color: #047857;">Healthy</span>
                    <a href="/docs" target="_blank">Swagger OpenAPI ↗</a>
                </div>
            </header>

            <div class="grid">
                <!-- Collections List -->
                <div class="card">
                    <h2>
                        <span>Active Collections</span>
                        <button onclick="fetchCollections()" style="padding: 0.35rem 0.75rem; font-size: 0.8rem;">Refresh</button>
                    </h2>
                    <div id="collections-list">Loading collections...</div>
                </div>

                <!-- Create Collection Form -->
                <div class="card">
                    <h2>Create New Collection</h2>
                    <div class="form-group">
                        <label>Collection Name</label>
                        <input id="create-name" placeholder="e.g. documents" value="demo_collection" />
                    </div>
                    <div class="form-group">
                        <label>Vector Dimension</label>
                        <input id="create-dim" type="number" placeholder="e.g. 4 or 1536" value="4" />
                    </div>
                    <div class="form-group" style="display: flex; gap: 1rem;">
                        <div style="flex: 1;">
                            <label>Distance Metric</label>
                            <select id="create-metric">
                                <option value="cosine">Cosine</option>
                                <option value="euclidean">Euclidean</option>
                                <option value="dot">Dot Product</option>
                            </select>
                        </div>
                        <div style="flex: 1;">
                            <label>Index Type</label>
                            <select id="create-index">
                                <option value="hnsw">HNSW (Graph ANN)</option>
                                <option value="flat">Flat (Exact)</option>
                            </select>
                        </div>
                    </div>
                    <button onclick="createCollection()">Create Collection</button>
                    <div id="create-status" class="status-msg"></div>
                </div>
            </div>

            <div class="grid">
                <!-- Insert Vector -->
                <div class="card">
                    <h2>Insert / Upsert Record</h2>
                    <div class="form-group">
                        <label>Target Collection</label>
                        <input id="insert-collection" value="demo_collection" />
                    </div>
                    <div class="form-group">
                        <label>Record ID</label>
                        <input id="insert-id" value="doc_1" />
                    </div>
                    <div class="form-group">
                        <label>Vector (Comma-separated floats)</label>
                        <input id="insert-vector" value="0.1, 0.4, 0.8, -0.3" />
                    </div>
                    <div class="form-group">
                        <label>Metadata JSON</label>
                        <textarea id="insert-metadata" rows="2">{"category": "tech", "author": "Alice"}</textarea>
                    </div>
                    <button onclick="insertRecord()">Insert Record</button>
                    <div id="insert-status" class="status-msg"></div>
                </div>

                <!-- Search Vectors -->
                <div class="card">
                    <h2>Vector Similarity Query</h2>
                    <div class="form-group">
                        <label>Collection</label>
                        <input id="search-collection" value="demo_collection" />
                    </div>
                    <div class="form-group">
                        <label>Query Vector (Comma-separated)</label>
                        <input id="search-vector" value="0.1, 0.35, 0.82, -0.28" />
                    </div>
                    <div class="form-group" style="display: flex; gap: 1rem;">
                        <div style="flex: 1;">
                            <label>Top K</label>
                            <input id="search-k" type="number" value="3" />
                        </div>
                        <div style="flex: 2;">
                            <label>Filter (Optional JSON)</label>
                            <input id="search-filter" placeholder='{"category": "tech"}' />
                        </div>
                    </div>
                    <button onclick="searchVectors()">Run Search</button>
                    <h3 style="font-size: 0.9rem; margin-top: 1rem; margin-bottom: 0.4rem; color: var(--text-muted);">Results:</h3>
                    <div id="search-results" class="results-box">No search performed yet.</div>
                </div>
            </div>
        </div>

        <script>
            function getHeaders() {
                const headers = {'Content-Type': 'application/json'};
                const key = localStorage.getItem('vdb_api_key') || '';
                if (key.trim()) headers['X-API-Key'] = key.trim();
                return headers;
            }

            function saveApiKey() {
                const key = document.getElementById('api-key-input').value;
                localStorage.setItem('vdb_api_key', key);
            }

            function selectCollection(name) {
                document.getElementById('insert-collection').value = name;
                document.getElementById('search-collection').value = name;
            }

            async function deleteCollection(name, event) {
                event.stopPropagation();
                if (!confirm(`Are you sure you want to delete collection '${name}'?`)) return;
                try {
                    const res = await fetch(`/api/v1/collections/${name}`, {
                        method: 'DELETE',
                        headers: getHeaders()
                    });
                    if (!res.ok) {
                        const err = await res.json();
                        alert('Error: ' + (err.detail || 'Failed'));
                    }
                    fetchCollections();
                } catch (e) {
                    alert('Error: ' + e.message);
                }
            }

            async function fetchCollections() {
                const listEl = document.getElementById('collections-list');
                try {
                    const res = await fetch('/api/v1/collections', { headers: getHeaders() });
                    const names = await res.json();
                    if (!Array.isArray(names) || names.length === 0) {
                        listEl.innerHTML = '<p style="color: var(--text-muted); font-size: 0.9rem;">No collections found. Create one on the right!</p>';
                        return;
                    }
                    let html = '';
                    for (const name of names) {
                        const statsRes = await fetch(`/api/v1/collections/${name}`, { headers: getHeaders() });
                        const stats = await statsRes.json();
                        html += `
                            <div class="collection-item" style="cursor: pointer;" onclick="selectCollection('${stats.name}')" title="Click to select for Insert/Search">
                                <div class="collection-info">
                                    <span class="collection-name">${stats.name} ↵</span>
                                    <span class="collection-meta">Dim: ${stats.dimension} | Metric: ${stats.metric} | Index: ${stats.index_type}</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 0.5rem;">
                                    <span class="badge">${stats.count} vectors</span>
                                    <button onclick="deleteCollection('${stats.name}', event)" style="background: #991b1b; padding: 0.25rem 0.55rem; font-size: 0.75rem;">Delete</button>
                                </div>
                            </div>
                        `;
                    }
                    listEl.innerHTML = html;
                } catch (e) {
                    listEl.innerHTML = '<p style="color: #ef4444;">Error loading collections: ' + e.message + '</p>';
                }
            }

            async function createCollection() {
                const status = document.getElementById('create-status');
                const name = document.getElementById('create-name').value;
                const dimension = parseInt(document.getElementById('create-dim').value);
                const metric = document.getElementById('create-metric').value;
                const index_type = document.getElementById('create-index').value;

                try {
                    const res = await fetch('/api/v1/collections', {
                        method: 'POST',
                        headers: getHeaders(),
                        body: JSON.stringify({ name, dimension, metric, index_type })
                    });
                    const data = await res.json();
                    if (!res.ok) throw new Error(data.detail || 'Failed');
                    status.innerHTML = `<span style="color: var(--success);">Created collection ${data.name}!</span>`;
                    selectCollection(data.name);
                    fetchCollections();
                } catch (e) {
                    status.innerHTML = `<span style="color: #ef4444;">${e.message}</span>`;
                }
            }

            async function insertRecord() {
                const status = document.getElementById('insert-status');
                const coll = document.getElementById('insert-collection').value;
                const id = document.getElementById('insert-id').value;
                const vecStr = document.getElementById('insert-vector').value;
                const metaStr = document.getElementById('insert-metadata').value;

                try {
                    const vector = vecStr.split(',').map(s => parseFloat(s.trim()));
                    const metadata = metaStr.trim() ? JSON.parse(metaStr) : {};
                    const res = await fetch(`/api/v1/collections/${coll}/insert`, {
                        method: 'POST',
                        headers: getHeaders(),
                        body: JSON.stringify({ id, vector, metadata })
                    });
                    const data = await res.json();
                    if (!res.ok) throw new Error(data.detail || 'Failed');
                    status.innerHTML = `<span style="color: var(--success);">${data.message}</span>`;
                    fetchCollections();
                } catch (e) {
                    status.innerHTML = `<span style="color: #ef4444;">${e.message}</span>`;
                }
            }

            async function searchVectors() {
                const resultsEl = document.getElementById('search-results');
                const coll = document.getElementById('search-collection').value;
                const vecStr = document.getElementById('search-vector').value;
                const k = parseInt(document.getElementById('search-k').value);
                const filterStr = document.getElementById('search-filter').value;

                try {
                    const vector = vecStr.split(',').map(s => parseFloat(s.trim()));
                    const filter = filterStr.trim() ? JSON.parse(filterStr) : null;
                    const res = await fetch(`/api/v1/collections/${coll}/search`, {
                        method: 'POST',
                        headers: getHeaders(),
                        body: JSON.stringify({ vector, k, filter })
                    });
                    const data = await res.json();
                    if (!res.ok) throw new Error(data.detail || 'Failed');
                    resultsEl.innerText = JSON.stringify(data, null, 2);
                } catch (e) {
                    resultsEl.innerText = 'Error: ' + e.message;
                }
            }

            // Restore saved API key and load collections
            const savedKey = localStorage.getItem('vdb_api_key');
            if (savedKey) document.getElementById('api-key-input').value = savedKey;
            fetchCollections();
        </script>
    </body>
    </html>
    """
