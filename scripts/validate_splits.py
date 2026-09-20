from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json

UNITS_PATH = Path(
    "data/derived/whole-child-generation-units.json"
)
QUESTIONS_PATH = Path(
    "artifacts/generated/"
    "whole-child-mixtral-questions.jsonl"
)
SPLIT_DIR = Path("data/splits/whole-child")
MANIFEST_PATH = SPLIT_DIR / "manifest.json"

SPLITS = ("train", "dev", "test")
EXPECTED_QUESTIONS = {
    "train": 1064,
    "dev": 392,
    "test": 1758,
}
EXPECTED_UNITS = {
    "train": 171,
    "dev": 88,
    "test": 296,
}
EXPECTED_LEAVES = {
    "train": 129,
    "dev": 66,
    "test": 86,
}
EXPECTED_TOTAL_QUESTIONS = 3214
EXPECTED_TOTAL_UNITS = 555


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def sha256(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


units = json.loads(
    UNITS_PATH.read_text(encoding="utf-8")
)
units_by_id = {
    unit["id"]: unit
    for unit in units
}
manifest = json.loads(
    MANIFEST_PATH.read_text(encoding="utf-8")
)

require(
    len(units_by_id) == EXPECTED_TOTAL_UNITS,
    "Unexpected generation-unit count",
)

rows_by_split = {
    split: read_jsonl(
        SPLIT_DIR / f"{split}.jsonl"
    )
    for split in SPLITS
}

actual_questions = {
    split: len(rows_by_split[split])
    for split in SPLITS
}
actual_units = {
    split: len(
        {
            row["source_unit_id"]
            for row in rows_by_split[split]
        }
    )
    for split in SPLITS
}
leaf_sets = {
    split: {
        row["gold_leaf_id"]
        for row in rows_by_split[split]
    }
    for split in SPLITS
}
actual_leaves = {
    split: len(leaf_sets[split])
    for split in SPLITS
}

require(
    actual_questions == EXPECTED_QUESTIONS,
    f"Question counts changed: {actual_questions}",
)
require(
    actual_units == EXPECTED_UNITS,
    f"Unit counts changed: {actual_units}",
)
require(
    actual_leaves == EXPECTED_LEAVES,
    f"Leaf coverage changed: {actual_leaves}",
)

all_rows = [
    row
    for split in SPLITS
    for row in rows_by_split[split]
]

require(
    len(all_rows) == EXPECTED_TOTAL_QUESTIONS,
    "Unexpected total question count",
)

question_id_counts = Counter(
    row["id"] for row in all_rows
)
duplicate_question_ids = sorted(
    question_id
    for question_id, count
    in question_id_counts.items()
    if count > 1
)
require(
    not duplicate_question_ids,
    (
        "Duplicate question IDs: "
        f"{duplicate_question_ids}"
    ),
)

normalized_query_counts = Counter(
    " ".join(row["query"].split()).casefold()
    for row in all_rows
)
duplicate_queries = sorted(
    query
    for query, count
    in normalized_query_counts.items()
    if count > 1
)
require(
    not duplicate_queries,
    (
        "Duplicate queries across splits: "
        f"{duplicate_queries}"
    ),
)

unit_splits = defaultdict(set)
observed_unit_ids = set()

for split in SPLITS:
    for row in rows_by_split[split]:
        require(
            row["split"] == split,
            f"Incorrect split field for {row['id']}",
        )

        unit_id = row["source_unit_id"]
        require(
            unit_id in units_by_id,
            f"Unknown source unit: {unit_id}",
        )

        unit = units_by_id[unit_id]

        require(
            row["gold_leaf_id"] == unit["leaf_id"],
            f"Leaf mismatch for {row['id']}",
        )
        require(
            row["gold_leaf_title"]
            == unit["leaf_title"],
            f"Leaf-title mismatch for {row['id']}",
        )
        require(
            row["gold_path"] == unit["path"],
            f"Path mismatch for {row['id']}",
        )
        require(
            row["source_paragraph_ids"]
            == unit["source_paragraph_ids"],
            (
                "Source-paragraph mismatch for "
                f"{row['id']}"
            ),
        )
        require(
            row["query"] == row["query"].strip()
            and row["query"].endswith("?"),
            f"Malformed query: {row['id']}",
        )

        unit_splits[unit_id].add(split)
        observed_unit_ids.add(unit_id)

require(
    observed_unit_ids == set(units_by_id),
    (
        "Missing units: "
        f"{sorted(set(units_by_id) - observed_unit_ids)}"
    ),
)

leaking_units = sorted(
    unit_id
    for unit_id, split_names
    in unit_splits.items()
    if len(split_names) != 1
)
require(
    not leaking_units,
    f"Units cross split boundaries: {leaking_units}",
)

units_per_leaf = Counter(
    unit["leaf_id"] for unit in units
)
all_leaves = set(units_per_leaf)
test_eligible = {
    leaf_id
    for leaf_id, count in units_per_leaf.items()
    if count >= 2
}
dev_eligible = {
    leaf_id
    for leaf_id, count in units_per_leaf.items()
    if count >= 3
}

require(
    leaf_sets["train"] == all_leaves,
    "Training does not contain every leaf",
)
require(
    leaf_sets["test"] == test_eligible,
    "Test coverage is not mathematically maximal",
)
require(
    leaf_sets["dev"] == dev_eligible,
    "Development coverage is not mathematically maximal",
)

repair_rows = [
    row
    for row in all_rows
    if row["model_generated"] is False
]
require(
    len(repair_rows) == 6,
    (
        "Expected six deterministic repair questions, "
        f"got {len(repair_rows)}"
    ),
)
require(
    all(
        row["split"] == "train"
        and row["split_constraint"] == "train_only"
        for row in repair_rows
    ),
    "Deterministic repairs escaped training",
)

for split in SPLITS:
    split_path = SPLIT_DIR / f"{split}.jsonl"
    require(
        manifest["split_sha256"][split]
        == sha256(split_path),
        f"Hash mismatch for {split}",
    )

require(
    manifest["actual_question_counts"]
    == EXPECTED_QUESTIONS,
    "Manifest question counts differ",
)
require(
    manifest["actual_unit_counts"]
    == EXPECTED_UNITS,
    "Manifest unit counts differ",
)
require(
    manifest["leaf_coverage"]
    == EXPECTED_LEAVES,
    "Manifest leaf coverage differs",
)
require(
    manifest["total_questions"]
    == EXPECTED_TOTAL_QUESTIONS,
    "Manifest total-question count differs",
)
require(
    manifest["total_units"]
    == EXPECTED_TOTAL_UNITS,
    "Manifest total-unit count differs",
)

source_verified = False

if QUESTIONS_PATH.exists():
    require(
        manifest["source_questions_sha256"]
        == sha256(QUESTIONS_PATH),
        "Source-question hash mismatch",
    )

    source_records = read_jsonl(QUESTIONS_PATH)
    expected_queries = {}

    for record in source_records:
        for index, question in enumerate(
            record["questions"],
            start=1,
        ):
            question_id = (
                f"{record['unit_id']}-q{index:02d}"
            )
            expected_queries[question_id] = question

    observed_queries = {
        row["id"]: row["query"]
        for row in all_rows
    }

    require(
        observed_queries == expected_queries,
        "Split questions differ from generated source",
    )
    source_verified = True

print("Split validation passed")
print("Question counts:", actual_questions)
print("Unit counts:", actual_units)
print("Leaf coverage:", actual_leaves)
print("Unit leakage: 0")
print("Global duplicate queries: 0")
print("Deterministic repair questions in train: 6")
print(
    "Generated source verified:",
    source_verified,
)
