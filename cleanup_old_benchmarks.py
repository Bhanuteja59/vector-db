import json
from pathlib import Path

meta_path = Path("data/televault_docs.meta.json")
snap_path = Path("data/televault_docs.snapshot.json")

if snap_path.exists():
    with open(snap_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta_store = data.get("metadata_store", {})
    vectors = data.get("vectors", {})
    node_layers = data.get("node_layers", {})

    print(f"Original records count: {len(vectors)}")

    # Keep only active valid documents ('lorem_ipsum_complete_pdf_doc', 'user_sample_pdf_001', 'doc_financial_q3', 'doc_technical_arch')
    keep_ids = set()
    for k, v in meta_store.items():
        fid = v.get("file_id")
        if fid in {'lorem_ipsum_complete_pdf_doc', 'user_sample_pdf_001', 'doc_financial_q3', 'doc_technical_arch'}:
            keep_ids.add(k)

    new_vectors = {k: v for k, v in vectors.items() if k in keep_ids}
    new_meta = {k: v for k, v in meta_store.items() if k in keep_ids}
    new_layers = {k: v for k, v in node_layers.items() if k in keep_ids}

    data["vectors"] = new_vectors
    data["metadata_store"] = new_meta
    data["node_layers"] = new_layers
    data["entry_point"] = list(keep_ids)[0] if keep_ids else None
    data["graphs"] = [{k: [n for n in v if n in keep_ids] for k, v in layer.items() if k in keep_ids} for layer in data.get("graphs", [])]

    with open(snap_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    print(f"Cleaned records count: {len(new_vectors)}. File size reduced from 100MB to {(snap_path.stat().st_size / 1024 / 1024):.2f} MB!")
