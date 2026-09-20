from collections import Counter, defaultdict
from pathlib import Path
import argparse
import json
import math

import numpy as np


DEFAULT_DSI = Path(
    "artifacts/results/whole-child-dsi-test.json"
)
DEFAULT_STAIR = Path(
    "artifacts/results/whole-child-stair-test.json"
)
DEFAULT_DATA_DIR = Path(
    "data/retriever/whole-child"
)
DEFAULT_OUTPUT = Path(
    "artifacts/results/whole-child-dsi-vs-stair-analysis.json"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsi", type=Path, default=DEFAULT_DSI)
    parser.add_argument("--stair", type=Path, default=DEFAULT_STAIR)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def exact_mcnemar_p_value(dsi_only, stair_only):
    discordant = dsi_only + stair_only

    if discordant == 0:
        return 1.0

    tail = min(dsi_only, stair_only)
    numerator = sum(
        math.comb(discordant, value)
        for value in range(tail + 1)
    )
    return min(1.0, 2.0 * numerator / (2**discordant))


def bootstrap_delta(dsi_correct, stair_correct, samples, seed):
    rng = np.random.default_rng(seed)
    paired_delta = stair_correct - dsi_correct
    estimates = np.empty(samples, dtype=np.float64)

    for index in range(samples):
        sample_indices = rng.integers(
            0,
            len(paired_delta),
            size=len(paired_delta),
        )
        estimates[index] = paired_delta[sample_indices].mean()

    return {
        "samples": samples,
        "seed": seed,
        "lower_95": float(np.quantile(estimates, 0.025)),
        "median": float(np.quantile(estimates, 0.5)),
        "upper_95": float(np.quantile(estimates, 0.975)),
    }


def metric_row(records):
    total = len(records)
    correct = sum(record["correct_at_1"] for record in records)
    return {
        "queries": total,
        "dsi_recall_at_1": (
            sum(record["dsi_correct_at_1"] for record in records)
            / total
        ),
        "stair_recall_at_1": (
            sum(record["stair_correct_at_1"] for record in records)
            / total
        ),
        "stair_minus_dsi": (
            sum(record["stair_correct_at_1"] for record in records)
            - sum(record["dsi_correct_at_1"] for record in records)
        ) / total,
        "combined_correct_at_1": correct / total,
    }


def main():
    args = parse_args()
    dsi_payload = json.loads(args.dsi.read_text(encoding="utf-8"))
    stair_payload = json.loads(args.stair.read_text(encoding="utf-8"))

    dsi_by_id = {
        prediction["id"]: prediction
        for prediction in dsi_payload["predictions"]
    }
    stair_by_id = {
        prediction["id"]: prediction
        for prediction in stair_payload["predictions"]
    }

    if set(dsi_by_id) != set(stair_by_id):
        raise ValueError("DSI and STAIR query IDs do not match")

    labels = json.loads(
        (args.data_dir / "labels.json").read_text(encoding="utf-8")
    )
    labels_by_id = {
        label["leaf_id"]: label
        for label in labels
    }
    train_records = read_jsonl(
        args.data_dir / "dsi" / "train.jsonl"
    )
    dev_records = read_jsonl(
        args.data_dir / "dsi" / "dev.jsonl"
    )
    train_counts = Counter(
        record["gold_leaf_id"]
        for record in train_records
    )
    dev_leaf_ids = {
        record["gold_leaf_id"]
        for record in dev_records
    }

    paired_records = []

    for query_id in sorted(dsi_by_id):
        dsi = dsi_by_id[query_id]
        stair = stair_by_id[query_id]

        for key in ("query", "gold_leaf_id", "source_unit_id"):
            if dsi[key] != stair[key]:
                raise ValueError(
                    f"Mismatched {key} for {query_id}"
                )

        dsi_correct = dsi["gold_rank_at_k"] == 1
        stair_correct = stair["gold_rank_at_k"] == 1
        leaf_id = dsi["gold_leaf_id"]
        label = labels_by_id[leaf_id]
        canonical_path = label.get("canonical_path", leaf_id)

        paired_records.append(
            {
                "id": query_id,
                "query": dsi["query"],
                "source_unit_id": dsi["source_unit_id"],
                "gold_leaf_id": leaf_id,
                "canonical_path": canonical_path,
                "hierarchy_depth": canonical_path.count(" > ") + 1,
                "train_query_count": train_counts[leaf_id],
                "leaf_seen_in_dev": leaf_id in dev_leaf_ids,
                "dsi_rank_at_5": dsi["gold_rank_at_k"],
                "stair_rank_at_5": stair["gold_rank_at_k"],
                "dsi_correct_at_1": dsi_correct,
                "stair_correct_at_1": stair_correct,
                "correct_at_1": dsi_correct or stair_correct,
                "outcome": (
                    "both_correct"
                    if dsi_correct and stair_correct
                    else "dsi_only"
                    if dsi_correct
                    else "stair_only"
                    if stair_correct
                    else "both_wrong"
                ),
                "dsi_top_1_leaf_id": dsi["ranked_targets"][0]["leaf_id"],
                "stair_top_1_leaf_id": stair["ranked_targets"][0]["leaf_id"],
            }
        )

    dsi_correct = np.array(
        [record["dsi_correct_at_1"] for record in paired_records],
        dtype=np.float64,
    )
    stair_correct = np.array(
        [record["stair_correct_at_1"] for record in paired_records],
        dtype=np.float64,
    )
    outcomes = Counter(record["outcome"] for record in paired_records)

    per_leaf = []

    for leaf_id in sorted({record["gold_leaf_id"] for record in paired_records}):
        group = [
            record
            for record in paired_records
            if record["gold_leaf_id"] == leaf_id
        ]
        row = metric_row(group)
        row.update(
            {
                "leaf_id": leaf_id,
                "canonical_path": group[0]["canonical_path"],
                "hierarchy_depth": group[0]["hierarchy_depth"],
                "train_query_count": group[0]["train_query_count"],
                "leaf_seen_in_dev": group[0]["leaf_seen_in_dev"],
            }
        )
        per_leaf.append(row)

    by_dev_coverage = {}

    for seen in (True, False):
        group = [
            record
            for record in paired_records
            if record["leaf_seen_in_dev"] is seen
        ]
        by_dev_coverage[
            "seen_in_dev" if seen else "absent_from_dev"
        ] = metric_row(group)

    by_depth = {}

    for depth in sorted({record["hierarchy_depth"] for record in paired_records}):
        group = [
            record
            for record in paired_records
            if record["hierarchy_depth"] == depth
        ]
        by_depth[str(depth)] = metric_row(group)

    dsi_recall = float(dsi_correct.mean())
    stair_recall = float(stair_correct.mean())
    bootstrap = bootstrap_delta(
        dsi_correct=dsi_correct,
        stair_correct=stair_correct,
        samples=args.bootstrap_samples,
        seed=args.seed,
    )

    summary = {
        "queries": len(paired_records),
        "dsi_recall_at_1": dsi_recall,
        "stair_recall_at_1": stair_recall,
        "stair_minus_dsi": stair_recall - dsi_recall,
        "relative_change": (
            (stair_recall - dsi_recall) / dsi_recall
        ),
        "outcomes": dict(outcomes),
        "mcnemar_exact_two_sided_p": exact_mcnemar_p_value(
            outcomes["dsi_only"],
            outcomes["stair_only"],
        ),
        "paired_bootstrap_stair_minus_dsi": bootstrap,
        "by_development_leaf_coverage": by_dev_coverage,
        "by_hierarchy_depth": by_depth,
    }

    payload = {
        "summary": summary,
        "per_leaf": sorted(
            per_leaf,
            key=lambda row: (
                row["stair_minus_dsi"],
                row["leaf_id"],
            ),
        ),
        "paired_predictions": paired_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Paired DSI versus STAIR analysis complete")
    print(f"Queries: {summary['queries']}")
    print(f"DSI Recall@1: {dsi_recall:.4f}")
    print(f"STAIR Recall@1: {stair_recall:.4f}")
    print(
        "STAIR - DSI: "
        f"{summary['stair_minus_dsi']:+.4f}"
    )
    print(f"Both correct: {outcomes['both_correct']}")
    print(f"DSI only: {outcomes['dsi_only']}")
    print(f"STAIR only: {outcomes['stair_only']}")
    print(f"Both wrong: {outcomes['both_wrong']}")
    print(
        "Exact McNemar p-value: "
        f"{summary['mcnemar_exact_two_sided_p']:.6g}"
    )
    print(
        "Paired bootstrap 95% CI: "
        f"[{bootstrap['lower_95']:+.4f}, "
        f"{bootstrap['upper_95']:+.4f}]"
    )

    for name, row in by_dev_coverage.items():
        print(
            f"{name}: n={row['queries']} | "
            f"DSI={row['dsi_recall_at_1']:.4f} | "
            f"STAIR={row['stair_recall_at_1']:.4f} | "
            f"delta={row['stair_minus_dsi']:+.4f}"
        )

    print("\nFive leaves favoring DSI most:")
    for row in payload["per_leaf"][:5]:
        print(
            f"- {row['stair_minus_dsi']:+.4f} | "
            f"n={row['queries']} | {row['canonical_path']}"
        )

    print("\nFive leaves favoring STAIR most:")
    for row in payload["per_leaf"][-5:][::-1]:
        print(
            f"- {row['stair_minus_dsi']:+.4f} | "
            f"n={row['queries']} | {row['canonical_path']}"
        )

    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
