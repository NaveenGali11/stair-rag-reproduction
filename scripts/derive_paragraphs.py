from collections import defaultdict
from pathlib import Path
import json
import re
import statistics

import pymupdf


PDF_PATH = Path("data/raw/whole-child.pdf")
NODES_PATH = Path("data/derived/whole-child-toc-nodes.json")
OUTPUT_PATH = Path("data/derived/whole-child-paragraphs.json")

TOP_MARGIN = 40
BOTTOM_MARGIN = 45
INDENT_THRESHOLD = 5
SHORT_LINE_THRESHOLD = 20

BULLET_PATTERN = re.compile(
    r"^(?:[•●▪◦*-]|\d+[.)]|[A-Za-z][.)])\s+"
)


def normalize(text: str) -> str:
    return " ".join(text.split())


def ends_sentence(text: str) -> bool:
    return bool(re.search(r"""[.!?][”"'’)\]]*$""", text))


def join_lines(lines: list[str]) -> str:
    result = ""

    for line in lines:
        if not result:
            result = line
        elif result.endswith("-"):
            result += line
        else:
            result += " " + line

    return normalize(result)


def locate_heading_blocks(document, nodes):
    page_blocks = {
        page_index: document[page_index].get_text("blocks", sort=True)
        for page_index in range(document.page_count)
    }

    heading_blocks = {}
    used_blocks = set()

    for node in nodes:
        page_index = node["page"] - 1
        blocks = page_blocks[page_index]
        title = normalize(node["title"])

        exact = [
            index
            for index, block in enumerate(blocks)
            if normalize(block[4]) == title
            and (page_index, index) not in used_blocks
        ]

        contained = [
            index
            for index, block in enumerate(blocks)
            if normalize(block[4]).startswith(title)
            and "|" in normalize(block[4])
            and (page_index, index) not in used_blocks
        ]

        candidates = exact or contained

        if not candidates:
            raise ValueError(
                f"Could not locate block for {node['id']}: "
                f"{node['title']}"
            )

        block_index = candidates[0]
        key = (page_index, block_index)

        heading_blocks[key] = node
        used_blocks.add(key)

    return page_blocks, heading_blocks


def main():
    nodes = json.loads(NODES_PATH.read_text(encoding="utf-8"))

    paragraphs = []
    paragraph_counts = defaultdict(int)

    active_leaf = None
    buffer = []
    previous_text = None
    previous_x1 = None
    previous_y1 = None

    def flush():
        nonlocal buffer
        nonlocal previous_text
        nonlocal previous_x1
        nonlocal previous_y1

        if not buffer or active_leaf is None:
            buffer = []
            previous_text = None
            previous_x1 = None
            previous_y1 = None
            return

        paragraph_counts[active_leaf["id"]] += 1
        number = paragraph_counts[active_leaf["id"]]
        text = join_lines([item["text"] for item in buffer])

        paragraphs.append(
            {
                "id": f"{active_leaf['id']}-p{number:03d}",
                "leaf_id": active_leaf["id"],
                "leaf_title": active_leaf["title"],
                "path": active_leaf["path"],
                "page_start": buffer[0]["page"],
                "page_end": buffer[-1]["page"],
                "text": text,
                "character_count": len(text),
            }
        )

        buffer = []
        previous_text = None
        previous_x1 = None
        previous_y1 = None

    with pymupdf.open(PDF_PATH) as document:
        page_blocks, heading_blocks = locate_heading_blocks(
            document,
            nodes,
        )

        for page_index in range(document.page_count):
            page = document[page_index]
            body_left = page.rect.width * (56.7 / 612.0)
            body_right = page.rect.width * (557.7 / 612.0)

            for block_index, block in enumerate(page_blocks[page_index]):
                heading = heading_blocks.get(
                    (page_index, block_index)
                )

                if heading is not None:
                    flush()
                    active_leaf = (
                        heading
                        if heading["is_retrieval_leaf"]
                        else None
                    )
                    continue

                x0, y0, x1, y1, raw_text = block[:5]
                text = normalize(raw_text)

                if not text:
                    continue

                if y1 <= TOP_MARGIN:
                    continue

                if y0 >= page.rect.height - BOTTOM_MARGIN:
                    continue

                if active_leaf is None:
                    continue

                starts_indented = x0 > body_left + INDENT_THRESHOLD
                previous_was_short = (
                    previous_x1 is not None
                    and previous_x1
                    < body_right - SHORT_LINE_THRESHOLD
                    and previous_text is not None
                    and ends_sentence(previous_text)
                )
                has_vertical_gap = (
                    previous_y1 is not None
                    and y0 > previous_y1
                    and y0 - previous_y1 > 6
                )
                starts_list_item = bool(BULLET_PATTERN.match(text))

                if buffer and (
                    starts_indented
                    or previous_was_short
                    or has_vertical_gap
                    or starts_list_item
                ):
                    flush()

                buffer.append(
                    {
                        "text": text,
                        "page": page_index + 1,
                    }
                )

                previous_text = text
                previous_x1 = x1
                previous_y1 = y1

        flush()

    expected_leaf_ids = {
        node["id"]
        for node in nodes
        if node["is_retrieval_leaf"]
    }
    observed_leaf_ids = {
        paragraph["leaf_id"] for paragraph in paragraphs
    }

    missing_leaf_ids = sorted(expected_leaf_ids - observed_leaf_ids)

    if missing_leaf_ids:
        raise ValueError(
            f"Leaves without paragraphs: {missing_leaf_ids}"
        )

    OUTPUT_PATH.write_text(
        json.dumps(paragraphs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lengths = [
        paragraph["character_count"] for paragraph in paragraphs
    ]

    print(f"Paragraphs: {len(paragraphs)}")
    print(f"Leaves represented: {len(observed_leaf_ids)}")
    print(f"Missing leaves: {len(missing_leaf_ids)}")
    print(
        "Median characters per paragraph: "
        f"{statistics.median(lengths):.0f}"
    )
    print(f"Shortest paragraph: {min(lengths)}")
    print(f"Longest paragraph: {max(lengths)}")
    print(f"Wrote: {OUTPUT_PATH}")

    example = [
        paragraph
        for paragraph in paragraphs
        if paragraph["leaf_id"] == "whole-child-023"
    ]

    print("\nwhole-child-023 paragraphs:")
    for paragraph in example:
        print(f"\n{paragraph['id']}:")
        print(paragraph["text"])


if __name__ == "__main__":
    main()