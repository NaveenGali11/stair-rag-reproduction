"""Validate the tracked release without raw data, models, or GPU packages."""

from __future__ import annotations

from pathlib import Path
import ast
import hashlib
import json
import re


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "dev", "test")
EXPECTED = {
    "toc_nodes": 181,
    "retrieval_leaves": 129,
    "paragraphs": 1019,
    "generation_units": 555,
    "split_questions": {"train": 1064, "dev": 392, "test": 1758},
    "split_units": {"train": 171, "dev": 88, "test": 296},
    "split_leaf_coverage": {"train": 129, "dev": 66, "test": 86},
}
EXPECTED_RECALL_AT_1 = {
    "bm25": 0.7986348122866894,
    "nvembed": 0.6552901023890785,
    "dsi": 0.40102389078498296,
    "stair": 0.37542662116040953,
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message: str):
    if not condition:
        raise AssertionError(message)


def validate_required_files():
    required = [
        "README.md",
        "RESULTS.md",
        "LICENSE",
        "DATA_LICENSE.md",
        "CITATION.cff",
        "CONTRIBUTING.md",
        "Makefile",
        "docs/REPRODUCING.md",
        "docs/BLOG.md",
        "results/whole-child-results.json",
        "notebooks/whole_child_results.ipynb",
    ]
    missing = [path for path in required if not (ROOT / path).is_file()]
    require(not missing, f"Missing release files: {missing}")


def validate_json_files():
    json_paths = sorted((ROOT / "data").rglob("*.json"))
    json_paths += sorted((ROOT / "results").rglob("*.json"))
    jsonl_paths = sorted((ROOT / "data").rglob("*.jsonl"))

    for path in json_paths:
        load_json(path)
    for path in jsonl_paths:
        load_jsonl(path)

    return len(json_paths), len(jsonl_paths)


def validate_corpus():
    derived = ROOT / "data/derived"
    nodes = load_json(derived / "whole-child-toc-nodes.json")
    sections = load_json(derived / "whole-child-sections.json")
    paragraphs = load_json(derived / "whole-child-paragraphs.json")
    units = load_json(derived / "whole-child-generation-units.json")
    leaves = {node["id"] for node in nodes if node["is_retrieval_leaf"]}

    require(len(nodes) == EXPECTED["toc_nodes"], "Unexpected ToC node count")
    require(
        len(leaves) == EXPECTED["retrieval_leaves"],
        "Unexpected retrieval-leaf count",
    )
    require(len(sections) == len(leaves), "Section/leaf count mismatch")
    require(
        len(paragraphs) == EXPECTED["paragraphs"],
        "Unexpected paragraph count",
    )
    require(
        len(units) == EXPECTED["generation_units"],
        "Unexpected generation-unit count",
    )
    require(
        {section["id"] for section in sections} == leaves,
        "Sections do not cover exactly the retrieval leaves",
    )
    require(
        {paragraph["leaf_id"] for paragraph in paragraphs} == leaves,
        "Paragraphs do not cover exactly the retrieval leaves",
    )
    require(
        {unit["leaf_id"] for unit in units} == leaves,
        "Generation units do not cover exactly the retrieval leaves",
    )
    return leaves


def validate_splits(leaves):
    split_dir = ROOT / "data/splits/whole-child"
    rows_by_split = {
        split: load_jsonl(split_dir / f"{split}.jsonl") for split in SPLITS
    }
    manifest = load_json(split_dir / "manifest.json")

    seen_ids = set()
    seen_queries = set()
    unit_to_split = {}
    observed_units = {}
    observed_coverage = {}

    for split, rows in rows_by_split.items():
        require(
            len(rows) == EXPECTED["split_questions"][split],
            f"Unexpected {split} question count",
        )
        units = {row["source_unit_id"] for row in rows}
        coverage = {row["gold_leaf_id"] for row in rows}
        observed_units[split] = len(units)
        observed_coverage[split] = len(coverage)

        for row in rows:
            require(row["split"] == split, f"Wrong split marker for {row['id']}")
            require(row["gold_leaf_id"] in leaves, f"Unknown gold leaf: {row['id']}")
            require(row["id"] not in seen_ids, f"Duplicate question ID: {row['id']}")
            require(
                row["query"] not in seen_queries,
                f"Duplicate query text: {row['id']}",
            )
            seen_ids.add(row["id"])
            seen_queries.add(row["query"])

            unit_id = row["source_unit_id"]
            previous = unit_to_split.setdefault(unit_id, split)
            require(previous == split, f"Source-unit leakage: {unit_id}")

            if row["split_constraint"] == "train_only":
                require(split == "train", f"Repair outside train: {row['id']}")
                require(not row["model_generated"], f"Repair mislabelled: {row['id']}")

    require(observed_units == EXPECTED["split_units"], "Unexpected split unit counts")
    require(
        observed_coverage == EXPECTED["split_leaf_coverage"],
        "Unexpected split leaf coverage",
    )
    require(
        set(unit_to_split) == {
            unit["id"]
            for unit in load_json(
                ROOT / "data/derived/whole-child-generation-units.json"
            )
        },
        "Split units do not cover exactly the generation units",
    )

    require(
        manifest["actual_question_counts"] == EXPECTED["split_questions"],
        "Split manifest question counts changed",
    )
    require(
        manifest["actual_unit_counts"] == EXPECTED["split_units"],
        "Split manifest unit counts changed",
    )
    require(
        manifest["leaf_coverage"] == EXPECTED["split_leaf_coverage"],
        "Split manifest leaf coverage changed",
    )

    for split in SPLITS:
        path = split_dir / f"{split}.jsonl"
        require(
            sha256(path) == manifest["split_sha256"][split],
            f"Split hash mismatch: {split}",
        )

    return rows_by_split


def validate_retriever_data(split_rows):
    data_dir = ROOT / "data/retriever/whole-child"
    labels = load_json(data_dir / "labels.json")
    manifest = load_json(data_dir / "manifest.json")
    target_by_leaf = {label["leaf_id"]: label["canonical_path"] for label in labels}

    require(len(labels) == EXPECTED["retrieval_leaves"], "Unexpected label count")
    require(
        manifest["counts"] == EXPECTED["split_questions"],
        "Retriever manifest counts changed",
    )

    for split in SPLITS:
        source = split_rows[split]
        source_by_id = {row["id"]: row for row in source}
        for method in ("dsi", "stair"):
            path = data_dir / method / f"{split}.jsonl"
            rows = load_jsonl(path)
            require(len(rows) == len(source), f"{method}/{split} count mismatch")
            require(
                sha256(path) == manifest["output_sha256"][str(path.relative_to(ROOT))],
                f"Retriever hash mismatch: {method}/{split}",
            )
            for row in rows:
                original = source_by_id.get(row["id"])
                require(original is not None, f"Unexpected retriever row: {row['id']}")
                require(row["query"] == original["query"], f"Query drift: {row['id']}")
                require(
                    row["gold_leaf_id"] == original["gold_leaf_id"],
                    f"Gold-label drift: {row['id']}",
                )
                expected_target = (
                    row["gold_leaf_id"]
                    if method == "dsi"
                    else target_by_leaf[row["gold_leaf_id"]]
                )
                require(row["target"] == expected_target, f"Bad target: {row['id']}")

    for relative, expected_hash in manifest["output_sha256"].items():
        require(sha256(ROOT / relative) == expected_hash, f"Hash mismatch: {relative}")


def validate_results():
    payload = load_json(ROOT / "results/whole-child-results.json")
    test_set = payload["test_set"]
    require(test_set["queries"] == 1758, "Frozen test query count changed")
    require(test_set["candidate_leaves"] == 129, "Candidate count changed")
    require(test_set["represented_gold_leaves"] == 86, "Gold coverage changed")

    for method, expected in EXPECTED_RECALL_AT_1.items():
        observed = payload["metrics"][method]["recall_at_1"]
        require(abs(observed - expected) < 1e-12, f"Recall@1 changed: {method}")

    paired = payload["paired_dsi_vs_stair"]
    require(paired["queries"] == 1758, "Paired query count changed")
    require(
        sum(paired["outcomes"].values()) == 1758,
        "Paired outcomes do not sum to the test count",
    )
    require(
        abs(paired["stair_minus_dsi"] + 0.025597269624573427) < 1e-12,
        "Paired delta changed",
    )


def validate_notebook():
    notebook = load_json(ROOT / "notebooks/whole_child_results.ipynb")
    require(notebook["nbformat"] == 4, "Unexpected notebook format")
    ids = [cell.get("id") for cell in notebook["cells"]]
    require(all(ids), "Notebook cell missing an ID")
    require(len(ids) == len(set(ids)), "Notebook cell IDs are not unique")

    code_cells = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        code_cells += 1
        ast.parse("".join(cell["source"]), filename=f"notebook-cell-{index}")
    require(code_cells == 7, "Unexpected notebook code-cell count")


def validate_relative_markdown_links():
    markdown_paths = [
        ROOT / "README.md",
        ROOT / "RESULTS.md",
        ROOT / "DATA_LICENSE.md",
        ROOT / "data/MANIFEST.md",
        ROOT / "docs/REPRODUCING.md",
        ROOT / "docs/BLOG.md",
    ]
    pattern = re.compile(r"!?(?:\[[^]]*\])\(([^)]+)\)")
    missing = []

    for path in markdown_paths:
        for target in pattern.findall(path.read_text(encoding="utf-8")):
            target = target.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{path.relative_to(ROOT)} -> {target}")

    require(not missing, f"Broken relative Markdown links: {missing}")


def main():
    validate_required_files()
    json_count, jsonl_count = validate_json_files()
    leaves = validate_corpus()
    split_rows = validate_splits(leaves)
    validate_retriever_data(split_rows)
    validate_results()
    validate_notebook()
    validate_relative_markdown_links()

    print("Repository validation passed")
    print(f"JSON/JSONL files: {json_count}/{jsonl_count}")
    print(f"ToC nodes: {EXPECTED['toc_nodes']}")
    print(f"Retrieval leaves: {EXPECTED['retrieval_leaves']}")
    print(f"Generation units: {EXPECTED['generation_units']}")
    print(f"Questions: {sum(EXPECTED['split_questions'].values())}")
    print("Frozen retrievers: BM25, NV-Embed-v2, DSI, STAIR")


if __name__ == "__main__":
    main()
