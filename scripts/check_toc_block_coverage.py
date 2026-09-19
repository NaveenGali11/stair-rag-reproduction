from pathlib import Path
import json

import pymupdf


PDF_PATH = Path("data/raw/whole-child.pdf")
NODES_PATH = Path("data/derived/whole-child-toc-nodes.json")


def normalize(text: str) -> str:
    return " ".join(text.split())


nodes = json.loads(NODES_PATH.read_text(encoding="utf-8"))

exact = []
contained = []
missing = []

with pymupdf.open(PDF_PATH) as document:
    for node in nodes:
        page = document[node["page"] - 1]
        blocks = [
            normalize(block[4])
            for block in page.get_text("blocks")
            if normalize(block[4])
        ]

        title = normalize(node["title"])

        exact_matches = [
            index
            for index, block_text in enumerate(blocks)
            if block_text == title
        ]

        contained_matches = [
            index
            for index, block_text in enumerate(blocks)
            if title in block_text
        ]

        if exact_matches:
            exact.append(node)
        elif contained_matches:
            contained.append(
                {
                    "node": node,
                    "blocks": [
                        blocks[index] for index in contained_matches
                    ],
                }
            )
        else:
            missing.append(node)

print(f"ToC nodes: {len(nodes)}")
print(f"Exact block matches: {len(exact)}")
print(f"Contained-only matches: {len(contained)}")
print(f"Missing block matches: {len(missing)}")

if contained:
    print("\nContained-only headings:")
    for item in contained:
        print(
            f"- {item['node']['id']} | "
            f"{item['node']['title']} | "
            f"{item['blocks']}"
        )

if missing:
    print("\nMissing headings:")
    for node in missing:
        print(
            f"- {node['id']} | PDF page {node['page']} | "
            f"{node['title']}"
        )