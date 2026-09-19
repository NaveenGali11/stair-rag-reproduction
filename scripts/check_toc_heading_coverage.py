from pathlib import Path
import json
import re

import pymupdf

pdf_path = Path("data/raw/whole-child.pdf")
nodes_path = Path("data/derived/whole-child-toc-nodes.json")

nodes = json.loads(nodes_path.read_text(encoding="utf-8"))


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


matched = []
missing = []

with pymupdf.open(pdf_path) as document:
    for node in nodes:
        page_index = node["page"] - 1
        page_text = document[page_index].get_text()

        if normalize(node["title"]) in normalize(page_text):
            matched.append(node)
        else:
            missing.append(node)

retrieval_leaves = [
    node for node in nodes if node["is_retrieval_leaf"]
]
missing_leaf_ids = {node["id"] for node in missing}
missing_leaves = [
    node for node in retrieval_leaves
    if node["id"] in missing_leaf_ids
]

print(f"All ToC headings found on declared page: {len(matched)}/{len(nodes)}")
print(
    "Retrieval-leaf headings found on declared page: "
    f"{len(retrieval_leaves) - len(missing_leaves)}/{len(retrieval_leaves)}"
)

if missing:
    print("\nFirst missing headings:")
    for node in missing[:20]:
        print(
            f"- {node['id']} | page {node['page']} | "
            f"{' > '.join(node['path'])}"
        )