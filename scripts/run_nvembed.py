import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel


MODEL_ID = "nvidia/NV-Embed-v2"
MODEL_REVISION = "3fa59658547db50a1e8e3346cf057fd0c77ed6ef"

PASSAGE_MAX_LENGTH = 512
QUERY_MAX_LENGTH = 256
PASSAGE_BATCH_SIZE = 4
QUERY_BATCH_SIZE = 4

QUERY_INSTRUCTION = (
    "Instruct: Given a question, retrieve passages that answer the question\n"
    "Query: "
)

SECTIONS_PATH = Path("data/derived/whole-child-sections.json")
QUERIES_PATH = Path("data/gold_queries.jsonl")
CACHE_PATH = Path("artifacts/embeddings/whole-child-nvembed-v2.npz")
RESULTS_PATH = Path("artifacts/results/whole-child-nvembed-v2.json")


def load_queries(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def corpus_fingerprint(sections: list[dict]) -> str:
    digest = hashlib.sha256()

    for section in sections:
        digest.update(section["id"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(section["text"].encode("utf-8"))
        digest.update(b"\0")

    return digest.hexdigest()


def normalize_embeddings(embeddings) -> torch.Tensor:
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings).cuda()

    return F.normalize(embeddings.float(), p=2, dim=1)


def encode_texts(
    model,
    texts: list[str],
    instruction: str,
    max_length: int,
    batch_size: int,
) -> torch.Tensor:
    with torch.inference_mode():
        embeddings = model._do_encode(
            texts,
            instruction=instruction,
            max_length=max_length,
            batch_size=batch_size,
            num_workers=0,
            return_numpy=False,
        )

    return normalize_embeddings(embeddings)


sections = json.loads(SECTIONS_PATH.read_text(encoding="utf-8"))
queries = load_queries(QUERIES_PATH)

if not queries:
    raise ValueError("No evaluation queries found")

section_ids = [section["id"] for section in sections]
section_by_id = {section["id"]: section for section in sections}
section_index = {
    section_id: index for index, section_id in enumerate(section_ids)
}

missing_gold_ids = sorted(
    {
        query["gold_leaf_id"]
        for query in queries
        if query["gold_leaf_id"] not in section_index
    }
)

if missing_gold_ids:
    raise ValueError(f"Missing gold leaf IDs: {missing_gold_ids}")

corpus_sha256 = corpus_fingerprint(sections)

print(f"Sections: {len(sections)}")
print(f"Queries: {len(queries)}")
print(f"Corpus SHA-256: {corpus_sha256}")
print(f"Loading {MODEL_ID} at revision {MODEL_REVISION}")

model = AutoModel.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
    attn_implementation="eager",
)
model.eval()

print(
    "GPU memory after model load: "
    f"{torch.cuda.memory_allocated() / 1024**3:.2f} GiB"
)

torch.cuda.reset_peak_memory_stats()

passage_embeddings = None

if CACHE_PATH.exists():
    with np.load(CACHE_PATH) as cache:
        cache_matches = (
            cache["model_revision"].item() == MODEL_REVISION
            and cache["corpus_sha256"].item() == corpus_sha256
            and cache["passage_max_length"].item() == PASSAGE_MAX_LENGTH
            and cache["section_ids"].tolist() == section_ids
        )

        if cache_matches:
            passage_embeddings = torch.from_numpy(
                cache["embeddings"]
            ).cuda()

            print(f"Loaded passage embeddings from {CACHE_PATH}")
        else:
            print("Existing cache metadata does not match; rebuilding")

if passage_embeddings is None:
    print("Encoding all passage sections")

    passage_embeddings = encode_texts(
        model=model,
        texts=[section["text"] for section in sections],
        instruction="",
        max_length=PASSAGE_MAX_LENGTH,
        batch_size=PASSAGE_BATCH_SIZE,
    )

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        CACHE_PATH,
        embeddings=passage_embeddings.cpu().numpy(),
        section_ids=np.asarray(section_ids),
        model_revision=np.asarray(MODEL_REVISION),
        corpus_sha256=np.asarray(corpus_sha256),
        passage_max_length=np.asarray(PASSAGE_MAX_LENGTH),
    )

    passage_embeddings = passage_embeddings.cuda()
    print(f"Cached passage embeddings at {CACHE_PATH}")

print("Encoding evaluation queries")

query_embeddings = encode_texts(
    model=model,
    texts=[query["query"] for query in queries],
    instruction=QUERY_INSTRUCTION,
    max_length=QUERY_MAX_LENGTH,
    batch_size=QUERY_BATCH_SIZE,
)

scores = query_embeddings @ passage_embeddings.T

ranks = []
query_results = []

for query_number, query in enumerate(queries):
    ranked_indices = torch.argsort(
        scores[query_number],
        descending=True,
    ).cpu().tolist()

    gold_index = section_index[query["gold_leaf_id"]]
    gold_rank = ranked_indices.index(gold_index) + 1
    ranks.append(gold_rank)

    print(f"\nQuery: {query['query']}")
    print(f"Gold: {query['gold_leaf_id']}")
    print(f"Gold rank: {gold_rank}")
    print("Top 5:")

    top_five = []

    for rank, candidate_index in enumerate(ranked_indices[:5], start=1):
        section = sections[candidate_index]
        score = scores[query_number, candidate_index].item() * 100
        marker = " <-- GOLD" if section["id"] == query["gold_leaf_id"] else ""
        path = " > ".join(section["path"])

        print(
            f"{rank}. {section['id']} | "
            f"{score:.3f} | {path}{marker}"
        )

        top_five.append(
            {
                "rank": rank,
                "leaf_id": section["id"],
                "score": score,
                "path": section["path"],
            }
        )

    query_results.append(
        {
            "id": query["id"],
            "query": query["query"],
            "gold_leaf_id": query["gold_leaf_id"],
            "gold_rank": gold_rank,
            "top_five": top_five,
        }
    )

query_count = len(queries)

metrics = {
    "recall_at_1": sum(rank <= 1 for rank in ranks) / query_count,
    "recall_at_3": sum(rank <= 3 for rank in ranks) / query_count,
    "recall_at_5": sum(rank <= 5 for rank in ranks) / query_count,
    "mrr": sum(1 / rank for rank in ranks) / query_count,
}

print("\nSummary")
print(f"Queries: {query_count}")
print(f"Recall@1: {metrics['recall_at_1']:.3f}")
print(f"Recall@3: {metrics['recall_at_3']:.3f}")
print(f"Recall@5: {metrics['recall_at_5']:.3f}")
print(f"MRR: {metrics['mrr']:.3f}")
print(
    "Peak GPU memory: "
    f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB"
)

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.write_text(
    json.dumps(
        {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "corpus_sha256": corpus_sha256,
            "section_count": len(sections),
            "query_count": query_count,
            "passage_max_length": PASSAGE_MAX_LENGTH,
            "query_max_length": QUERY_MAX_LENGTH,
            "metrics": metrics,
            "queries": query_results,
        },
        indent=2,
    ),
    encoding="utf-8",
)

print(f"Wrote results to {RESULTS_PATH}")