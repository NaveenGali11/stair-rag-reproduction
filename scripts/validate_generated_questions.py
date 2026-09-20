from collections import Counter
from pathlib import Path
import json

UNITS_PATH = Path(
    "data/derived/whole-child-generation-units.json"
)
REMOVALS_PATH = Path(
    "data/derived/whole-child-generation-removals.json"
)
QUESTIONS_PATH = Path(
    "artifacts/generated/"
    "whole-child-mixtral-questions.jsonl"
)
FAILURES_PATH = Path(
    "artifacts/generated/"
    "whole-child-mixtral-questions.failures.jsonl"
)

EXPECTED_UNITS = 555
EXPECTED_LEAVES = 129
EXPECTED_QUESTIONS = 3214
EXPECTED_PLACEHOLDER_IDS = {
    "whole-child-055-p016",
    "whole-child-109-p006",
}

REQUIRED_RECORD_KEYS = {
    "unit_id",
    "leaf_id",
    "source_paragraph_ids",
    "requested_question_count",
    "questions",
    "model_id",
    "model_revision",
    "prompt_version",
    "seed",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    if not path.exists():
        return []

    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


units = json.loads(
    UNITS_PATH.read_text(encoding="utf-8")
)
removals = json.loads(
    REMOVALS_PATH.read_text(encoding="utf-8")
)
records = read_jsonl(QUESTIONS_PATH)
failure_records = read_jsonl(FAILURES_PATH)

require(
    len(units) == EXPECTED_UNITS,
    f"Expected {EXPECTED_UNITS} units, got {len(units)}",
)

unit_ids = [unit["id"] for unit in units]
unit_id_counts = Counter(unit_ids)
duplicate_unit_ids = sorted(
    unit_id
    for unit_id, count in unit_id_counts.items()
    if count > 1
)

require(
    not duplicate_unit_ids,
    f"Duplicate generation-unit IDs: {duplicate_unit_ids}",
)

units_by_id = {
    unit["id"]: unit
    for unit in units
}
expected_ids = set(units_by_id)

placeholder_units = [
    unit["id"]
    for unit in units
    if "excluded from this version of the text"
    in unit["text"].casefold()
]

require(
    not placeholder_units,
    f"Interactive placeholders remain: {placeholder_units}",
)

placeholder_removals = {
    removal["source_paragraph_id"]
    for removal in removals
    if removal["reason"]
    == "interactive_element_placeholder"
}

require(
    placeholder_removals
    == EXPECTED_PLACEHOLDER_IDS,
    (
        "Unexpected interactive-placeholder removals: "
        f"{sorted(placeholder_removals)}"
    ),
)

record_ids = [
    record["unit_id"]
    for record in records
]
record_id_counts = Counter(record_ids)
duplicate_record_ids = sorted(
    unit_id
    for unit_id, count in record_id_counts.items()
    if count > 1
)
valid_ids = set(record_ids)

require(
    not duplicate_record_ids,
    f"Duplicate question-record IDs: {duplicate_record_ids}",
)
require(
    valid_ids == expected_ids,
    (
        f"Missing={sorted(expected_ids - valid_ids)}; "
        f"unexpected={sorted(valid_ids - expected_ids)}"
    ),
)

all_questions = []

for record in records:
    missing_keys = (
        REQUIRED_RECORD_KEYS - set(record)
    )
    require(
        not missing_keys,
        (
            f"{record.get('unit_id')} missing keys: "
            f"{sorted(missing_keys)}"
        ),
    )

    unit = units_by_id[record["unit_id"]]

    require(
        record["leaf_id"] == unit["leaf_id"],
        f"Leaf mismatch for {record['unit_id']}",
    )
    require(
        record["source_paragraph_ids"]
        == unit["source_paragraph_ids"],
        (
            "Source-paragraph mismatch for "
            f"{record['unit_id']}"
        ),
    )

    questions = record["questions"]
    requested = record["requested_question_count"]

    require(
        isinstance(questions, list),
        f"Questions are not a list for {record['unit_id']}",
    )
    require(
        len(questions) == requested,
        (
            f"Question-count mismatch for "
            f"{record['unit_id']}: "
            f"{len(questions)} != {requested}"
        ),
    )

    local_normalized = []

    for question in questions:
        require(
            isinstance(question, str),
            (
                "Non-string question in "
                f"{record['unit_id']}"
            ),
        )
        require(
            question == question.strip(),
            (
                "Untrimmed question in "
                f"{record['unit_id']}: {question!r}"
            ),
        )
        require(
            question.endswith("?"),
            (
                "Question lacks question mark in "
                f"{record['unit_id']}: {question!r}"
            ),
        )

        normalized = " ".join(
            question.split()
        ).casefold()
        local_normalized.append(normalized)
        all_questions.append(normalized)

    require(
        len(local_normalized)
        == len(set(local_normalized)),
        (
            "Duplicate questions within "
            f"{record['unit_id']}"
        ),
    )

require(
    len(records) == EXPECTED_UNITS,
    f"Expected {EXPECTED_UNITS} records, got {len(records)}",
)
require(
    len(all_questions) == EXPECTED_QUESTIONS,
    (
        f"Expected {EXPECTED_QUESTIONS} questions, "
        f"got {len(all_questions)}"
    ),
)

leaf_ids = {
    record["leaf_id"]
    for record in records
}
require(
    len(leaf_ids) == EXPECTED_LEAVES,
    (
        f"Expected {EXPECTED_LEAVES} leaves, "
        f"got {len(leaf_ids)}"
    ),
)

global_duplicates = sorted(
    question
    for question, count
    in Counter(all_questions).items()
    if count > 1
)
require(
    not global_duplicates,
    (
        "Global exact duplicate questions: "
        f"{global_duplicates}"
    ),
)

repair_records = [
    record
    for record in records
    if record.get("model_generated") is False
]
require(
    len(repair_records) == 1,
    (
        "Expected one deterministic repair record, "
        f"got {len(repair_records)}"
    ),
)
require(
    repair_records[0].get("split_constraint")
    == "train_only",
    "Deterministic repair is not constrained to train",
)

failure_ids = {
    record["unit_id"]
    for record in failure_records
}
unresolved_failures = sorted(
    (failure_ids & expected_ids) - valid_ids
)
require(
    not unresolved_failures,
    (
        "Unresolved expected failure units: "
        f"{unresolved_failures}"
    ),
)

print("Generated-question validation passed")
print(f"Expected units: {len(expected_ids)}")
print(f"Valid records: {len(records)}")
print(f"Leaf coverage: {len(leaf_ids)}")
print(f"Total questions: {len(all_questions)}")
print(
    "Deterministic repair records: "
    f"{len(repair_records)}"
)
print(
    "Failure-attempt records: "
    f"{len(failure_records)}"
)
print("Unresolved expected failure units: 0")
print("Global exact duplicate groups: 0")
