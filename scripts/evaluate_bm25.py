from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import time

from rank_bm25 import BM25Okapi


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sections",
        type=Path,
        default=Path(
            "data/derived/whole-child-sections.json"
        ),
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path(
            "data/splits/whole-child/test.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/results/"
            "whole-child-bm25-test.json"
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    return parser.parse_args()


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


def tokenize(text):
    return re.findall(
        r"\b\w+\b",
        text.casefold(),
    )


def mean(values):
    return sum(values) / len(values)


def metrics_from_ranks(ranks):
    return {
        "recall_at_1": mean(
            [rank <= 1 for rank in ranks]
        ),
        "recall_at_3": mean(
            [rank <= 3 for rank in ranks]
        ),
        "recall_at_5": mean(
            [rank <= 5 for rank in ranks]
        ),
        "mrr": mean(
            [1.0 / rank for rank in ranks]
        ),
        "ndcg_at_3": mean(
            [
                (
                    1.0 / math.log2(rank + 1)
                    if rank <= 3
                    else 0.0
                )
                for rank in ranks
            ]
        ),
        "mean_rank": mean(ranks),
        "median_rank": sorted(ranks)[
            len(ranks) // 2
        ],
    }


def main():
    args = parse_args()

    sections = json.loads(
        args.sections.read_text(
            encoding="utf-8"
        )
    )
    queries = read_jsonl(args.queries)

    if not sections:
        raise ValueError("No sections found")

    if not queries:
        raise ValueError("No queries found")

    section_ids = {
        section["id"]
        for section in sections
    }
    query_ids = [
        query["id"]
        for query in queries
    ]

    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Duplicate query IDs")

    for query in queries:
        if query["gold_leaf_id"] not in section_ids:
            raise ValueError(
                "Unknown gold leaf "
                f"{query['gold_leaf_id']} "
                f"for query {query['id']}"
            )

    tokenized_corpus = [
        tokenize(section["text"])
        for section in sections
    ]
    bm25 = BM25Okapi(tokenized_corpus)

    ranks = []
    ranks_by_leaf = defaultdict(list)
    results = []

    start = time.perf_counter()

    for query in queries:
        scores = bm25.get_scores(
            tokenize(query["query"])
        )

        ranked_indices = sorted(
            range(len(sections)),
            key=lambda index: (
                -float(scores[index]),
                sections[index]["id"],
            ),
        )

        gold_rank = next(
            rank
            for rank, index in enumerate(
                ranked_indices,
                start=1,
            )
            if sections[index]["id"]
            == query["gold_leaf_id"]
        )

        ranks.append(gold_rank)
        ranks_by_leaf[
            query["gold_leaf_id"]
        ].append(gold_rank)

        predictions = []

        for rank, index in enumerate(
            ranked_indices[: args.top_k],
            start=1,
        ):
            section = sections[index]
            predictions.append(
                {
                    "rank": rank,
                    "leaf_id": section["id"],
                    "title": section["title"],
                    "path": section["path"],
                    "score": float(scores[index]),
                    "is_gold": (
                        section["id"]
                        == query["gold_leaf_id"]
                    ),
                }
            )

        results.append(
            {
                "query_id": query["id"],
                "query": query["query"],
                "gold_leaf_id": query[
                    "gold_leaf_id"
                ],
                "gold_leaf_title": query[
                    "gold_leaf_title"
                ],
                "gold_rank": gold_rank,
                "top_predictions": predictions,
            }
        )

    elapsed = time.perf_counter() - start
    micro_metrics = metrics_from_ranks(ranks)

    per_leaf_metrics = {
        leaf_id: metrics_from_ranks(
            leaf_ranks
        )
        for leaf_id, leaf_ranks
        in sorted(ranks_by_leaf.items())
    }

    macro_metrics = {
        metric: mean(
            [
                leaf_metrics[metric]
                for leaf_metrics
                in per_leaf_metrics.values()
            ]
        )
        for metric in (
            "recall_at_1",
            "recall_at_3",
            "recall_at_5",
            "mrr",
            "ndcg_at_3",
            "mean_rank",
        )
    }

    payload = {
        "retriever": "BM25Okapi",
        "rank_bm25_version": version(
            "rank-bm25"
        ),
        "tokenization": (
            r"casefold + regex \b\w+\b"
        ),
        "tie_break": "ascending leaf ID",
        "sections_path": str(args.sections),
        "sections_sha256": sha256(
            args.sections
        ),
        "queries_path": str(args.queries),
        "queries_sha256": sha256(
            args.queries
        ),
        "query_count": len(queries),
        "candidate_count": len(sections),
        "evaluated_leaf_count": len(
            ranks_by_leaf
        ),
        "retrieval_seconds": elapsed,
        "milliseconds_per_query": (
            elapsed / len(queries) * 1000
        ),
        "micro_metrics": micro_metrics,
        "macro_leaf_metrics": macro_metrics,
        "per_leaf_metrics": per_leaf_metrics,
        "results": results,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("BM25 evaluation complete")
    print(f"Queries: {len(queries)}")
    print(f"Candidates: {len(sections)}")
    print(
        "Evaluated leaves: "
        f"{len(ranks_by_leaf)}"
    )
    print(
        "Recall@1: "
        f"{micro_metrics['recall_at_1']:.4f}"
    )
    print(
        "Recall@3: "
        f"{micro_metrics['recall_at_3']:.4f}"
    )
    print(
        "Recall@5: "
        f"{micro_metrics['recall_at_5']:.4f}"
    )
    print(
        "MRR: "
        f"{micro_metrics['mrr']:.4f}"
    )
    print(
        "nDCG@3: "
        f"{micro_metrics['ndcg_at_3']:.4f}"
    )
    print(
        "Macro leaf Recall@1: "
        f"{macro_metrics['recall_at_1']:.4f}"
    )
    print(
        "Milliseconds/query: "
        f"{payload['milliseconds_per_query']:.3f}"
    )
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
