from pathlib import Path
import json
import re
import statistics

import pymupdf

pdf_path = Path("data/raw/whole-child.pdf")
nodes_path = Path("data/derived/whole-child-toc-nodes.json")
output_path = Path("data/derived/whole-child-sections.json")

nodes = json.loads(nodes_path.read_text(encoding="utf-8"))


def heading_pattern(title: str) -> str:
    return re.escape(title).replace(r"\ ", r"\s+")


def find_heading(text: str, title: str):
    exact_line = re.compile(
        rf"(?m)^\s*{heading_pattern(title)}\s*$"
    ).search(text)

    if exact_line:
        return exact_line

    return re.compile(heading_pattern(title)).search(text)


def text_until_next_heading(page_texts, current, following):
    if following is None:
        parts = [page_texts[current["page_index"]][current["heading_end"] :]]
        parts.extend(page_texts[current["page_index"] + 1 :])
        return "\n".join(parts).strip()

    if current["page_index"] == following["page_index"]:
        return page_texts[current["page_index"]][
            current["heading_end"] : following["heading_start"]
        ].strip()

    parts = [page_texts[current["page_index"]][current["heading_end"] :]]
    parts.extend(
        page_texts[current["page_index"] + 1 : following["page_index"]]
    )
    parts.append(
        page_texts[following["page_index"]][: following["heading_start"]]
    )
    return "\n".join(parts).strip()


with pymupdf.open(pdf_path) as document:
    page_texts = [page.get_text() for page in document]

located_nodes = []

for node in nodes:
    page_index = node["page"] - 1
    match = find_heading(page_texts[page_index], node["title"])

    if match is None:
        raise ValueError(
            f"Could not locate {node['id']} on PDF page {node['page']}"
        )

    located_nodes.append(
        {
            **node,
            "page_index": page_index,
            "heading_start": match.start(),
            "heading_end": match.end(),
        }
    )

sections = []

for index, node in enumerate(located_nodes):
    if not node["is_retrieval_leaf"]:
        continue

    following = (
        located_nodes[index + 1]
        if index + 1 < len(located_nodes)
        else None
    )
    text = text_until_next_heading(page_texts, node, following)

    sections.append(
        {
            "id": node["id"],
            "title": node["title"],
            "path": node["path"],
            "page": node["page"],
            "text": text,
            "character_count": len(text),
        }
    )

empty_sections = [section for section in sections if not section["text"]]

if empty_sections:
    raise ValueError(f"Empty sections: {empty_sections}")

output_path.write_text(
    json.dumps(sections, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

lengths = [section["character_count"] for section in sections]
example = next(
    section for section in sections
    if section["id"] == "whole-child-023"
)

print(f"Extracted retrieval sections: {len(sections)}")
print(f"Empty sections: {len(empty_sections)}")
print(f"Median characters per section: {statistics.median(lengths):.0f}")
print(f"Wrote: {output_path}")
print(f"\nExample: {' > '.join(example['path'])}\n")
print(example["text"][:800])