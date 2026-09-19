from collections import Counter, defaultdict
from pathlib import Path
import json
import re
import statistics


NODES_PATH = Path(
    "data/derived/whole-child-toc-nodes.json"
)
PARAGRAPHS_PATH = Path(
    "data/derived/whole-child-paragraphs.json"
)

EXPECTED_LEAF_COUNT = 129
EXPECTED_PARAGRAPH_COUNT = 1019

CHAPTER_HEADING_PATTERN = re.compile(
    r"\bCHAPTER (?:ONE|TWO|THREE|FOUR|FIVE|SIX|"
    r"SEVEN|EIGHT|NINE)\b"
)


nodes = json.loads(
    NODES_PATH.read_text(encoding="utf-8")
)
paragraphs = json.loads(
    PARAGRAPHS_PATH.read_text(encoding="utf-8")
)

expected_nodes = {
    node["id"]: node
    for node in nodes
    if node["is_retrieval_leaf"]
}

expected_leaf_ids = set(expected_nodes)
observed_leaf_ids = {
    paragraph["leaf_id"]
    for paragraph in paragraphs
}

errors = []

if len(expected_leaf_ids) != EXPECTED_LEAF_COUNT:
    errors.append(
        "Expected "
        f"{EXPECTED_LEAF_COUNT} retrieval leaves, found "
        f"{len(expected_leaf_ids)}"
    )

if len(paragraphs) != EXPECTED_PARAGRAPH_COUNT:
    errors.append(
        "Expected "
        f"{EXPECTED_PARAGRAPH_COUNT} paragraphs, found "
        f"{len(paragraphs)}"
    )

missing_leaf_ids = sorted(
    expected_leaf_ids - observed_leaf_ids
)
extra_leaf_ids = sorted(
    observed_leaf_ids - expected_leaf_ids
)

if missing_leaf_ids:
    errors.append(
        f"Leaves without paragraphs: {missing_leaf_ids}"
    )

if extra_leaf_ids:
    errors.append(
        f"Unexpected paragraph leaves: {extra_leaf_ids}"
    )

paragraph_id_counts = Counter(
    paragraph["id"]
    for paragraph in paragraphs
)

duplicate_ids = sorted(
    paragraph_id
    for paragraph_id, count
    in paragraph_id_counts.items()
    if count > 1
)

if duplicate_ids:
    errors.append(
        f"Duplicate paragraph IDs: {duplicate_ids}"
    )

paragraphs_by_leaf = defaultdict(list)

for paragraph in paragraphs:
    paragraphs_by_leaf[paragraph["leaf_id"]].append(
        paragraph
    )

    text = paragraph["text"]

    if not text.strip():
        errors.append(
            f"Empty paragraph: {paragraph['id']}"
        )

    if paragraph["character_count"] != len(text):
        errors.append(
            "Character-count mismatch: "
            f"{paragraph['id']}"
        )

    if paragraph["page_start"] > paragraph["page_end"]:
        errors.append(
            f"Invalid page range: {paragraph['id']}"
        )

    node = expected_nodes.get(paragraph["leaf_id"])

    if node is None:
        continue

    if paragraph["leaf_title"] != node["title"]:
        errors.append(
            f"Leaf-title mismatch: {paragraph['id']}"
        )

    if paragraph["path"] != node["path"]:
        errors.append(
            f"Path mismatch: {paragraph['id']}"
        )

for leaf_id, group in paragraphs_by_leaf.items():
    expected_ids = [
        f"{leaf_id}-p{index:03d}"
        for index in range(1, len(group) + 1)
    ]
    actual_ids = [
        paragraph["id"]
        for paragraph in group
    ]

    if actual_ids != expected_ids:
        errors.append(
            f"Non-sequential paragraph IDs: {leaf_id}"
        )

chapter_leaks = [
    paragraph["id"]
    for paragraph in paragraphs
    if CHAPTER_HEADING_PATTERN.search(
        paragraph["text"]
    )
]

objective_leaks = [
    paragraph["id"]
    for paragraph in paragraphs
    if "After completing Chapter" in paragraph["text"]
]

conclusion_leaks = [
    paragraph["id"]
    for paragraph in paragraphs
    if paragraph["text"].strip() == "CONCLUSION"
]

if chapter_leaks:
    errors.append(
        f"Leaked chapter headings: {chapter_leaks}"
    )

if objective_leaks:
    errors.append(
        f"Leaked chapter objectives: {objective_leaks}"
    )

if conclusion_leaks:
    errors.append(
        f"Leaked conclusion heading: {conclusion_leaks}"
    )

if "whole-child-121" in observed_leaf_ids:
    errors.append(
        "Media Attributions leaf is present"
    )

spinal_text = " ".join(
    paragraph["text"]
    for paragraph in paragraphs_by_leaf[
        "whole-child-084"
    ]
).casefold()

medulla_text = " ".join(
    paragraph["text"]
    for paragraph in paragraphs_by_leaf[
        "whole-child-085"
    ]
).casefold()

if (
    "the spinal cord connects the brain "
    "to the rest of the body."
    not in spinal_text
):
    errors.append(
        "Spinal Cord regression check failed"
    )

if "medulla oblongata controls" not in medulla_text:
    errors.append(
        "Medulla Oblongata regression check failed"
    )

if errors:
    print("Paragraph validation failed:\n")

    for error in errors:
        print(f"- {error}")

    raise SystemExit(1)

lengths = [
    paragraph["character_count"]
    for paragraph in paragraphs
]

print("Paragraph validation passed")
print(f"Expected leaves: {len(expected_leaf_ids)}")
print(f"Represented leaves: {len(observed_leaf_ids)}")
print(f"Paragraphs: {len(paragraphs)}")
print(
    "Median characters per paragraph: "
    f"{statistics.median(lengths):.0f}"
)
print(f"Shortest paragraph: {min(lengths)}")
print(f"Longest paragraph: {max(lengths)}")