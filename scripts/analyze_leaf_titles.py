from collections import defaultdict
from pathlib import Path
import json

leaf_path = Path("data/derived/whole-child-retrieval-leaves.json")
leaves = json.loads(leaf_path.read_text(encoding="utf-8"))

by_title = defaultdict(list)

for leaf in leaves:
    by_title[leaf["title"]].append(leaf)

duplicates = {
    title: entries
    for title, entries in by_title.items()
    if len(entries) > 1
}

print(f"Retrieval leaves: {len(leaves)}")
print(f"Unique leaf titles: {len(by_title)}")
print(f"Repeated leaf titles: {len(duplicates)}\n")

for title, entries in duplicates.items():
    print(f"{title!r} occurs {len(entries)} times:")
    for entry in entries:
        print(f"  - {' > '.join(entry['path'])}")