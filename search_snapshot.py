import json

with open("data/televault_docs.snapshot.json", "r", encoding="utf-8") as f:
    data = json.load(f)

meta_store = data.get("metadata_store", {})

matches = []
for k, v in meta_store.items():
    if v.get("file_id") == "8d5abf4d-af6c-4b24-9872-b91db2ab6de7":
        txt = v.get("text_snippet", "")
        if "CyberShield" in txt or "Quantum" in txt or "Failover" in txt or "NEEDLE" in txt:
            matches.append((k, v.get("chunk_index"), txt))

print(f"Total matching chunks found in snapshot for file 8d5abf4d: {len(matches)}")
for m in matches:
    print(f"Chunk {m[1]}: {m[2]}")
