from pathlib import Path
import json
import re

from rank_bm25 import BM25Okapi

sections_path = Path("data/derived/whole-child-sections.json")
queries_path = Path("data/gold_queries.jsonl")

sections = json.loads(sections_path.read_text(encoding="utf-8"))

queries = [
    json.loads(line)
    for line in queries_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.casefold())


section_ids = {section["id"] for section in sections}

for query in queries:
    if query["gold_leaf_id"] not in section_ids:
        raise ValueError(
            f"Unknown gold leaf {query['gold_leaf_id']} "
            f"for query {query['id']}"
        )

tokenized_corpus = [
    tokenize(section["text"])
    for section in sections
]

bm25 = BM25Okapi(tokenized_corpus)
ranks = []

for query in queries:
    scores = bm25.get_scores(tokenize(query["query"]))

    ranked_indices = sorted(
        range(len(sections)),
        key=lambda index: (-float(scores[index]), sections[index]["id"]),
    )

    gold_rank = next(
        rank
        for rank, index in enumerate(ranked_indices, start=1)
        if sections[index]["id"] == query["gold_leaf_id"]
    )
    ranks.append(gold_rank)

    print(f"\nQuery: {query['query']}")
    print(f"Gold: {query['gold_leaf_id']}")
    print(f"Gold rank: {gold_rank}")
    print("Top 5:")

    for rank, index in enumerate(ranked_indices[:5], start=1):
        section = sections[index]
        marker = " <-- GOLD" if section["id"] == query["gold_leaf_id"] else ""
        print(
            f"{rank}. {section['id']} | {scores[index]:.3f} | "
            f"{' > '.join(section['path'])}{marker}"
        )

recall_at_1 = sum(rank <= 1 for rank in ranks) / len(ranks)
recall_at_3 = sum(rank <= 3 for rank in ranks) / len(ranks)

print("\nSummary")
print(f"Queries: {len(ranks)}")
print(f"Recall@1: {recall_at_1:.3f}")
print(f"Recall@3: {recall_at_3:.3f}")