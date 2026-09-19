from pathlib import Path
import json

import pymupdf

pdf_path = Path("data/raw/whole-child.pdf")
output_path = Path("data/derived/whole-child-toc-nodes.json")

NON_CONTENT_LEAF_TITLES = {
    "media attributions",
}

with pymupdf.open(pdf_path) as document:
    raw_toc = document.get_toc(simple=True)

nodes = []
stack = []

for position, (level, title, page) in enumerate(raw_toc, start=1):
    while stack and stack[-1]["level"] >= level:
        stack.pop()

    parent = stack[-1] if stack else None
    path = [*parent["path"], title] if parent else [title]

    node = {
        "id": f"whole-child-{position:03d}",
        "level": level,
        "title": title,
        "page": page,
        "parent_id": parent["id"] if parent else None,
        "path": path,
    }

    nodes.append(node)
    stack.append(node)

parent_ids = {
    node["parent_id"]
    for node in nodes
    if node["parent_id"] is not None
}

for node in nodes:
    node["is_leaf"] = node["id"] not in parent_ids

leaves = [node for node in nodes if node["is_leaf"]]

chapter_leaves = [
    node
    for node in leaves
    if node["path"][0].startswith("Chapter ")
    and node["title"].strip().casefold()
    not in NON_CONTENT_LEAF_TITLES
]

chapter_leaf_ids = {node["id"] for node in chapter_leaves}

for node in nodes:
    node["is_retrieval_leaf"] = node["id"] in chapter_leaf_ids

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(nodes, indent=2), encoding="utf-8")
leaf_output_path = Path("data/derived/whole-child-retrieval-leaves.json")
leaf_output_path.write_text(
    json.dumps(chapter_leaves, indent=2),
    encoding="utf-8",
)


print(f"All ToC nodes: {len(nodes)}")
print(f"Leaf nodes before filtering: {len(leaves)}")
print(f"Leaves under numbered chapters: {len(chapter_leaves)}")
print(f"Wrote: {output_path}\n")

for node in chapter_leaves[:40]:
    print(f"{node['id']} | page {node['page']:>3} | {' > '.join(node['path'])}")