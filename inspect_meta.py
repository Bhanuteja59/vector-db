import json

with open("data/televault_docs.snapshot.json", "r", encoding="utf-8") as f:
    data = json.load(f)

meta_store = data.get("metadata_store", {})
print(f"Total metadata records in snapshot: {len(meta_store)}")

file_ids = set()
for k, v in meta_store.items():
    file_ids.add(v.get("file_id"))

print(f"Unique file_ids in AegisVector DB snapshot: {file_ids}")
