from pathlib import Path
import hashlib
import json
import subprocess


RESULT_FILES = {
    "bm25": Path("artifacts/results/whole-child-bm25-test.json"),
    "nvembed": Path(
        "artifacts/results/whole-child-nvembed-v2-test.json"
    ),
    "dsi": Path("artifacts/results/whole-child-dsi-test.json"),
    "stair": Path("artifacts/results/whole-child-stair-test.json"),
    "paired_analysis": Path(
        "artifacts/results/whole-child-dsi-vs-stair-analysis.json"
    ),
}

ADAPTER_FILES = {
    "dsi_config": Path(
        "artifacts/models/dsi/best-adapter/adapter_config.json"
    ),
    "dsi_weights": Path(
        "artifacts/models/dsi/best-adapter/adapter_model.safetensors"
    ),
    "stair_config": Path(
        "artifacts/models/stair/best-adapter/adapter_config.json"
    ),
    "stair_weights": Path(
        "artifacts/models/stair/best-adapter/adapter_model.safetensors"
    ),
}

TRAINING_FILES = {
    "dsi_run_config": Path("artifacts/models/dsi/run-config.json"),
    "dsi_state": Path("artifacts/models/dsi/state.json"),
    "stair_run_config": Path("artifacts/models/stair/run-config.json"),
    "stair_state": Path("artifacts/models/stair/state.json"),
}

OUTPUT_JSON = Path("results/whole-child-results.json")
OUTPUT_MARKDOWN = Path("RESULTS.md")

EXPECTED = {
    "bm25": {
        "recall_at_1": 0.7986,
        "recall_at_3": 0.9107,
        "recall_at_5": 0.9357,
        "ndcg_at_3": 0.8648,
    },
    "nvembed": {
        "recall_at_1": 0.6553,
        "recall_at_3": 0.8271,
        "recall_at_5": 0.8697,
        "ndcg_at_3": 0.7553,
    },
    "dsi": {
        "recall_at_1": 0.4010,
        "recall_at_3": 0.5228,
        "recall_at_5": 0.5830,
        "ndcg_at_3": 0.4716,
    },
    "stair": {
        "recall_at_1": 0.3754,
        "recall_at_3": 0.4812,
        "recall_at_5": 0.4994,
        "ndcg_at_3": 0.4373,
    },
}


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def artifact_record(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def first_number(mapping, *keys, default=None):
    for key in keys:
        value = mapping.get(key)

        if isinstance(value, (int, float)):
            return float(value)

    if default is not None:
        return float(default)

    raise KeyError(f"None of these metrics were found: {keys}")


def normalized_summary(payload, method):
    if method in {"bm25", "nvembed"}:
        source = payload["micro_metrics"]
        result = {
            "queries": int(payload["query_count"]),
            "candidates": int(
                payload.get("candidate_count", payload.get("section_count"))
            ),
            "evaluated_leaves": int(payload["evaluated_leaf_count"]),
            "macro_leaf_recall_at_1": float(
                payload["macro_leaf_metrics"]["recall_at_1"]
            ),
            "milliseconds_per_query": float(
                payload[
                    "milliseconds_per_query"
                    if method == "bm25"
                    else "milliseconds_per_query_encoding"
                ]
            ),
        }
    else:
        source = payload["summary"]
        result = {
            "queries": int(source["queries"]),
            "candidates": int(source["candidates"]),
            "evaluated_leaves": int(source["evaluated_leaves"]),
            "macro_leaf_recall_at_1": float(
                source["macro_leaf_recall_at_1"]
            ),
            "milliseconds_per_query": float(
                source["milliseconds_per_query"]
            ),
        }

    result.update({
        "recall_at_1": first_number(
            source,
            "recall_at_1",
            "recall@1",
        ),
        "recall_at_3": first_number(
            source,
            "recall_at_3",
            "recall@3",
        ),
        "recall_at_5": first_number(
            source,
            "recall_at_5",
            "recall@5",
        ),
        "ndcg_at_3": first_number(
            source,
            "ndcg_at_3",
            "ndcg@3",
        ),
    })

    if method in {"dsi", "stair"}:
        result["mrr_at_5"] = first_number(
            source,
            "mrr_at_5",
        )
        result["peak_gpu_gib"] = first_number(
            source,
            "peak_gpu_gib",
        )
    else:
        result["mrr"] = first_number(source, "mrr")

        if method == "nvembed":
            result["peak_gpu_gib"] = float(
                payload["peak_gpu_memory_gib"]
            )

    return result


def validate_metrics(method, summary):
    if summary["queries"] != 1758:
        raise ValueError(f"{method}: expected 1758 queries")

    if summary["candidates"] != 129:
        raise ValueError(f"{method}: expected 129 candidates")

    if summary["evaluated_leaves"] != 86:
        raise ValueError(f"{method}: expected 86 evaluated leaves")

    for key, expected in EXPECTED[method].items():
        actual = summary[key]

        if abs(actual - expected) > 0.000051:
            raise ValueError(
                f"{method} {key}: {actual} does not round to {expected}"
            )


def markdown_table(metrics):
    rows = [
        "| Retriever | Recall@1 | Recall@3 | Recall@5 | "
        "nDCG@3 | Macro leaf Recall@1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    names = {
        "bm25": "BM25",
        "nvembed": "NV-Embed-v2",
        "dsi": "DSI",
        "stair": "STAIR",
    }

    for method in ("bm25", "nvembed", "dsi", "stair"):
        row = metrics[method]
        rows.append(
            f"| {names[method]} | "
            f"{row['recall_at_1']:.4f} | "
            f"{row['recall_at_3']:.4f} | "
            f"{row['recall_at_5']:.4f} | "
            f"{row['ndcg_at_3']:.4f} | "
            f"{row['macro_leaf_recall_at_1']:.4f} |"
        )

    return "\n".join(rows)


def main():
    payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in RESULT_FILES.items()
    }
    metrics = {
        method: normalized_summary(payloads[method], method)
        for method in ("bm25", "nvembed", "dsi", "stair")
    }

    for method, summary in metrics.items():
        validate_metrics(method, summary)

    paired_summary = payloads["paired_analysis"]["summary"]

    if paired_summary["queries"] != 1758:
        raise ValueError("Paired analysis query count is not 1758")

    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()
    training = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in TRAINING_FILES.items()
    }
    artifacts = {
        name: artifact_record(path)
        for name, path in {
            **RESULT_FILES,
            **ADAPTER_FILES,
            **TRAINING_FILES,
        }.items()
    }

    output = {
        "experiment": "Whole Child independent STAIR reproduction",
        "git_commit": git_commit,
        "test_set": {
            "queries": 1758,
            "candidate_leaves": 129,
            "represented_gold_leaves": 86,
            "grouped_by_source_unit": True,
        },
        "metrics": metrics,
        "paired_dsi_vs_stair": paired_summary,
        "training": training,
        "artifacts": artifacts,
        "interpretation": {
            "primary_result": (
                "STAIR did not outperform DSI overall on the frozen test set."
            ),
            "statistical_result": (
                "The 2.56-point DSI advantage is borderline: exact "
                "McNemar p=0.0542525 and the paired-bootstrap 95% "
                "interval for STAIR minus DSI is [-0.0506, 0.0000]."
            ),
            "depth_interaction": (
                "STAIR leads by 0.2644 Recall@1 on depth-2 queries "
                "and trails by 0.1061 on depth-3 queries."
            ),
        },
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    markdown = f"""# Whole Child retrieval results

These are the frozen results from an independent, one-book reproduction of
STAIR. All four retrievers use the same leakage-safe 1,758-query test split,
129 candidate leaves, and 86 represented gold leaves.

{markdown_table(metrics)}

BM25 and NV-Embed report full-candidate MRR. DSI and STAIR report MRR@5
because their constrained decoder retains five ranked beams.

## DSI versus STAIR

DSI reaches {metrics['dsi']['recall_at_1']:.4f} Recall@1 and STAIR reaches
{metrics['stair']['recall_at_1']:.4f}, a STAIR-minus-DSI difference of
{paired_summary['stair_minus_dsi']:+.4f}. The exact paired McNemar p-value is
{paired_summary['mcnemar_exact_two_sided_p']:.6f}. The paired-bootstrap 95%
interval is [{paired_summary['paired_bootstrap_stair_minus_dsi']['lower_95']:+.4f},
{paired_summary['paired_bootstrap_stair_minus_dsi']['upper_95']:+.4f}].

The aggregate result hides a strong depth interaction. STAIR improves
Recall@1 by 0.2644 on depth-2 queries but trails DSI by 0.1061 on depth-3
queries. This independent reproduction therefore does not reproduce the
paper's aggregate STAIR-over-DSI ordering, while still finding substantial
benefit from explicit structure for coarse routing.

Machine-readable metrics, training state, and SHA-256 artifact checksums are
stored in `results/whole-child-results.json`.
"""
    OUTPUT_MARKDOWN.write_text(markdown, encoding="utf-8")

    print("Frozen result manifest created")
    print(f"Git commit: {git_commit}")
    print(f"Wrote: {OUTPUT_JSON}")
    print(f"Wrote: {OUTPUT_MARKDOWN}")


if __name__ == "__main__":
    main()
