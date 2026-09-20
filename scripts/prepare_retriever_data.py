from pathlib import Path
import hashlib
import json

NODES_PATH = Path(
    "data/derived/whole-child-toc-nodes.json"
)
SPLIT_DIR = Path("data/splits/whole-child")
OUTPUT_DIR = Path("data/retriever/whole-child")

SPLITS = ("train", "dev", "test")

DSI_PROMPT_TEMPLATE = (
    "Retrieve the relevant document section for "
    "the query. Return only its identifier.\n\n"
    "Query: {query}"
)

STAIR_PROMPT_TEMPLATE = (
    "Select the single table-of-contents leaf "
    "that best answers the query. Return only "
    "its complete hierarchical path exactly as "
    "it appears in the table of contents.\n\n"
    "Table of Contents:\n{toc}\n\n"
    "Query: {query}"
)


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def write_jsonl(path, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def sha256(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


nodes = json.loads(
    NODES_PATH.read_text(encoding="utf-8")
)
retrieval_leaves = [
    node
    for node in nodes
    if node["is_retrieval_leaf"]
]

if len(retrieval_leaves) != 129:
    raise ValueError(
        "Expected 129 retrieval leaves, got "
        f"{len(retrieval_leaves)}"
    )

toc = "\n".join(
    (
        "  " * (node["level"] - 1)
        + "- "
        + node["title"]
    )
    for node in nodes
)

labels = []

for leaf in retrieval_leaves:
    canonical_path = " > ".join(
        leaf["path"]
    )
    labels.append(
        {
            "leaf_id": leaf["id"],
            "title": leaf["title"],
            "path": leaf["path"],
            "canonical_path": canonical_path,
        }
    )

label_by_id = {
    label["leaf_id"]: label
    for label in labels
}

canonical_paths = [
    label["canonical_path"]
    for label in labels
]

if len(canonical_paths) != len(
    set(canonical_paths)
):
    raise ValueError(
        "Canonical retrieval paths are not unique"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

toc_path = OUTPUT_DIR / "toc.txt"
labels_path = OUTPUT_DIR / "labels.json"

toc_path.write_text(
    toc + "\n",
    encoding="utf-8",
)
labels_path.write_text(
    json.dumps(
        labels,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)

counts = {}
output_paths = []

for split in SPLITS:
    source_path = (
        SPLIT_DIR / f"{split}.jsonl"
    )
    source_rows = read_jsonl(source_path)

    dsi_rows = []
    stair_rows = []

    for row in source_rows:
        leaf_id = row["gold_leaf_id"]

        if leaf_id not in label_by_id:
            raise ValueError(
                f"Unknown leaf ID: {leaf_id}"
            )

        label = label_by_id[leaf_id]

        if row["gold_path"] != label["path"]:
            raise ValueError(
                f"Path mismatch for {row['id']}"
            )

        common = {
            "id": row["id"],
            "query": row["query"],
            "gold_leaf_id": leaf_id,
            "source_unit_id": (
                row["source_unit_id"]
            ),
            "model_generated": (
                row["model_generated"]
            ),
            "split_constraint": (
                row["split_constraint"]
            ),
        }

        dsi_rows.append(
            {
                **common,
                "target": leaf_id,
            }
        )
        stair_rows.append(
            {
                **common,
                "target": label[
                    "canonical_path"
                ],
            }
        )

    dsi_path = (
        OUTPUT_DIR / "dsi" / f"{split}.jsonl"
    )
    stair_path = (
        OUTPUT_DIR
        / "stair"
        / f"{split}.jsonl"
    )

    write_jsonl(dsi_path, dsi_rows)
    write_jsonl(stair_path, stair_rows)

    output_paths.extend(
        [dsi_path, stair_path]
    )
    counts[split] = len(source_rows)

manifest = {
    "version": "whole-child-retriever-data-v1",
    "source_nodes_path": str(NODES_PATH),
    "source_nodes_sha256": sha256(
        NODES_PATH
    ),
    "source_split_manifest_path": str(
        SPLIT_DIR / "manifest.json"
    ),
    "source_split_manifest_sha256": sha256(
        SPLIT_DIR / "manifest.json"
    ),
    "toc_path": str(toc_path),
    "toc_sha256": sha256(toc_path),
    "labels_path": str(labels_path),
    "labels_sha256": sha256(labels_path),
    "toc_node_count": len(nodes),
    "retrieval_leaf_count": len(labels),
    "counts": counts,
    "targets": {
        "dsi": "opaque leaf ID",
        "stair": "unique complete hierarchical path",
    },
    "prompt_templates": {
        "dsi": DSI_PROMPT_TEMPLATE,
        "stair": STAIR_PROMPT_TEMPLATE,
    },
    "maximum_sequence_lengths": {
        "dsi": 512,
        "stair": 2048,
        "generation": 64,
    },
    "output_sha256": {
        str(path): sha256(path)
        for path in output_paths
    },
}

manifest_path = OUTPUT_DIR / "manifest.json"
manifest_path.write_text(
    json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)

print("Retriever data preparation passed")
print(f"ToC nodes: {len(nodes)}")
print(f"Retrieval leaves: {len(labels)}")
print(f"Unique STAIR targets: {len(set(canonical_paths))}")
print("Split counts:", counts)
print(f"ToC characters: {len(toc)}")
print(f"Wrote: {OUTPUT_DIR}")
