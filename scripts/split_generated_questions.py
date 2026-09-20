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
OUTPUT_DIR = Path("data/splits/whole-child")
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

SEED = 42
ALGORITHM_VERSION = "grouped-leaf-aware-v1"

PAPER_SPLIT_COUNTS = {
    "train": 1056,
    "dev": 388,
    "test": 1746,
}
PAPER_TOTAL = sum(PAPER_SPLIT_COUNTS.values())
SPLITS = ("train", "dev", "test")


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


def stable_key(value):
    return hashlib.sha256(
        f"{SEED}:{value}".encode("utf-8")
    ).hexdigest()


def proportional_targets(total):
    targets = {
        split: (
            total * PAPER_SPLIT_COUNTS[split]
            // PAPER_TOTAL
        )
        for split in SPLITS
    }

    remaining = total - sum(targets.values())

    remainders = sorted(
        SPLITS,
        key=lambda split: (
            -(
                total * PAPER_SPLIT_COUNTS[split]
                % PAPER_TOTAL
            ),
            SPLITS.index(split),
        ),
    )

    for split in remainders[:remaining]:
        targets[split] += 1

    return targets


units = json.loads(
    UNITS_PATH.read_text(encoding="utf-8")
)
records = read_jsonl(QUESTIONS_PATH)

units_by_id = {
    unit["id"]: unit
    for unit in units
}
records_by_id = {
    record["unit_id"]: record
    for record in records
}

if set(units_by_id) != set(records_by_id):
    raise ValueError(
        "Generation units and question records differ"
    )

total_questions = sum(
    len(record["questions"])
    for record in records
)
targets = proportional_targets(total_questions)

records_by_leaf = defaultdict(list)

for record in records:
    records_by_leaf[record["leaf_id"]].append(
        record
    )

for leaf_records in records_by_leaf.values():
    leaf_records.sort(
        key=lambda record: stable_key(
            record["unit_id"]
        )
    )

assignments = {}
question_counts = Counter()
unit_counts = Counter()
leaf_split_units = defaultdict(Counter)


def assign(record, split):
    unit_id = record["unit_id"]

    if unit_id in assignments:
        raise ValueError(
            f"Unit assigned twice: {unit_id}"
        )

    assignments[unit_id] = split
    question_counts[split] += len(
        record["questions"]
    )
    unit_counts[split] += 1
    leaf_split_units[
        record["leaf_id"]
    ][split] += 1


# Honor explicit split constraints first.
for record in records:
    constraint = record.get("split_constraint")

    if constraint is None:
        continue

    if constraint != "train_only":
        raise ValueError(
            f"Unknown split constraint: {constraint}"
        )

    assign(record, "train")


# Guarantee that every retrieval label is learned.
for leaf_id in sorted(records_by_leaf):
    leaf_records = records_by_leaf[leaf_id]

    if leaf_split_units[leaf_id]["train"]:
        continue

    candidate = next(
        record
        for record in leaf_records
        if record["unit_id"] not in assignments
    )
    assign(candidate, "train")


# Give every leaf with at least two units one
# independent test unit. This maximizes test-label
# coverage after reserving training coverage.
for leaf_id in sorted(records_by_leaf):
    candidates = [
        record
        for record in records_by_leaf[leaf_id]
        if record["unit_id"] not in assignments
    ]

    if candidates:
        assign(candidates[0], "test")


# Give every leaf with at least three units one
# independent development unit. Prefer the smallest
# remaining group to preserve the global dev target.
for leaf_id in sorted(records_by_leaf):
    candidates = [
        record
        for record in records_by_leaf[leaf_id]
        if record["unit_id"] not in assignments
    ]

    if candidates:
        candidate = min(
            candidates,
            key=lambda record: (
                len(record["questions"]),
                stable_key(record["unit_id"]),
            ),
        )
        assign(candidate, "dev")


# Interleave leaves so large leaves do not dominate
# consecutive allocation decisions.
remaining_by_leaf = {}

for leaf_id, leaf_records in records_by_leaf.items():
    remaining_by_leaf[leaf_id] = [
        record
        for record in leaf_records
        if record["unit_id"] not in assignments
    ]

leaf_order = sorted(
    remaining_by_leaf,
    key=stable_key,
)

interleaved = []
depth = 0

while True:
    added = False

    for leaf_id in leaf_order:
        queue = remaining_by_leaf[leaf_id]

        if depth < len(queue):
            interleaved.append(queue[depth])
            added = True

    if not added:
        break

    depth += 1


def split_score(record, split):
    target = targets[split]
    deficit_ratio = (
        target - question_counts[split]
    ) / target

    leaf_spread = int(
        leaf_split_units[
            record["leaf_id"]
        ][split] == 0
    )

    # Global count fidelity is primary; leaf spread
    # resolves near-ties.
    return (
        round(deficit_ratio, 6),
        leaf_spread,
        -SPLITS.index(split),
    )


for record in interleaved:
    split = max(
        SPLITS,
        key=lambda name: split_score(
            record,
            name,
        ),
    )
    assign(record, split)


if set(assignments) != set(records_by_id):
    raise ValueError("Not every unit was assigned")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rows_by_split = {
    split: []
    for split in SPLITS
}

for unit_id in sorted(
    records_by_id,
    key=stable_key,
):
    record = records_by_id[unit_id]
    unit = units_by_id[unit_id]
    split = assignments[unit_id]

    for index, question in enumerate(
        record["questions"],
        start=1,
    ):
        rows_by_split[split].append(
            {
                "id": (
                    f"{unit_id}-q{index:02d}"
                ),
                "query": question,
                "gold_leaf_id": record["leaf_id"],
                "gold_leaf_title": unit["leaf_title"],
                "gold_path": unit["path"],
                "split": split,
                "source_unit_id": unit_id,
                "source_paragraph_ids": (
                    unit["source_paragraph_ids"]
                ),
                "model_generated": record.get(
                    "model_generated",
                    True,
                ),
                "split_constraint": record.get(
                    "split_constraint"
                ),
                "generation_model_id": record[
                    "model_id"
                ],
                "generation_model_revision": record[
                    "model_revision"
                ],
                "generation_prompt_version": record[
                    "prompt_version"
                ],
                "generation_seed": record["seed"],
            }
        )

for split in SPLITS:
    rows_by_split[split].sort(
        key=lambda row: row["id"]
    )

    output_path = OUTPUT_DIR / f"{split}.jsonl"
    output_path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
            for row in rows_by_split[split]
        ),
        encoding="utf-8",
    )

actual_questions = {
    split: len(rows_by_split[split])
    for split in SPLITS
}
actual_units = {
    split: unit_counts[split]
    for split in SPLITS
}
leaf_coverage = {
    split: len(
        {
            row["gold_leaf_id"]
            for row in rows_by_split[split]
        }
    )
    for split in SPLITS
}

manifest = {
    "algorithm_version": ALGORITHM_VERSION,
    "seed": SEED,
    "group_key": "source_unit_id",
    "source_questions_path": str(
        QUESTIONS_PATH
    ),
    "source_questions_sha256": sha256(
        QUESTIONS_PATH
    ),
    "source_units_path": str(UNITS_PATH),
    "source_units_sha256": sha256(UNITS_PATH),
    "paper_reference_counts": (
        PAPER_SPLIT_COUNTS
    ),
    "target_question_counts": targets,
    "actual_question_counts": (
        actual_questions
    ),
    "actual_unit_counts": actual_units,
    "leaf_coverage": leaf_coverage,
    "split_sha256": {
        split: sha256(
            OUTPUT_DIR / f"{split}.jsonl"
        )
        for split in SPLITS
    },
    "total_questions": total_questions,
    "total_units": len(units),
    "total_leaves": len(records_by_leaf),
    "constraints": {
        "all_questions_from_unit_share_split": True,
        "all_leaves_represented_in_train": True,
        "deterministic_repairs_train_only": True,
    },
}

MANIFEST_PATH.write_text(
    json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)

print("Split generation complete")
print("Question targets:", targets)
print("Question counts:", actual_questions)
print("Unit counts:", actual_units)
print("Leaf coverage:", leaf_coverage)
print(f"Wrote: {OUTPUT_DIR}")
