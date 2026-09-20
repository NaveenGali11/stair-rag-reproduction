from collections import defaultdict
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel


MODEL_ID = "nvidia/NV-Embed-v2"
MODEL_REVISION = (
    "3fa59658547db50a1e8e3346cf057fd0c77ed6ef"
)

PASSAGE_MAX_LENGTH = 512
QUERY_MAX_LENGTH = 256
PASSAGE_BATCH_SIZE = 4
QUERY_BATCH_SIZE = 16

QUERY_INSTRUCTION = (
    "Instruct: Given a question, retrieve passages "
    "that answer the question\n"
    "Query: "
)

SECTIONS_PATH = Path(
    "data/derived/whole-child-sections.json"
)
QUERIES_PATH = Path(
    "data/splits/whole-child/test.jsonl"
)
CACHE_PATH = Path(
    "artifacts/embeddings/"
    "whole-child-nvembed-v2.npz"
)
RESULTS_PATH = Path(
    "artifacts/results/"
    "whole-child-nvembed-v2-test.json"
)


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def file_sha256(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def corpus_fingerprint(sections):
    digest = hashlib.sha256()

    for section in sections:
        digest.update(
            section["id"].encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(
            section["text"].encode("utf-8")
        )
        digest.update(b"\0")

    return digest.hexdigest()


def normalize_embeddings(embeddings):
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(
            embeddings
        ).cuda()

    return F.normalize(
        embeddings.float(),
        p=2,
        dim=1,
    )


def encode_texts(
    model,
    texts,
    instruction,
    max_length,
    batch_size,
):
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


def mean(values):
    return sum(values) / len(values)


def metrics_from_ranks(ranks):
    sorted_ranks = sorted(ranks)

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
        "median_rank": sorted_ranks[
            len(sorted_ranks) // 2
        ],
    }


sections = json.loads(
    SECTIONS_PATH.read_text(encoding="utf-8")
)
queries = read_jsonl(QUERIES_PATH)

if not sections:
    raise ValueError("No sections found")

if not queries:
    raise ValueError("No evaluation queries found")

query_ids = [query["id"] for query in queries]

if len(query_ids) != len(set(query_ids)):
    raise ValueError("Duplicate query IDs")

section_ids = [
    section["id"] for section in sections
]
section_index = {
    section_id: index
    for index, section_id
    in enumerate(section_ids)
}

missing_gold_ids = sorted(
    {
        query["gold_leaf_id"]
        for query in queries
        if query["gold_leaf_id"]
        not in section_index
    }
)

if missing_gold_ids:
    raise ValueError(
        f"Missing gold leaf IDs: {missing_gold_ids}"
    )

corpus_sha256 = corpus_fingerprint(sections)
queries_sha256 = file_sha256(QUERIES_PATH)

print(f"Sections: {len(sections)}")
print(f"Queries: {len(queries)}")
print(f"Corpus SHA-256: {corpus_sha256}")
print(
    f"Loading {MODEL_ID} "
    f"at revision {MODEL_REVISION}"
)

model_load_start = time.perf_counter()

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

model_load_seconds = (
    time.perf_counter() - model_load_start
)

print(
    "Model load seconds: "
    f"{model_load_seconds:.2f}"
)
print(
    "GPU memory after model load: "
    f"{torch.cuda.memory_allocated() / 1024**3:.2f} GiB"
)

torch.cuda.reset_peak_memory_stats()

passage_embeddings = None
passage_cache_hit = False

if CACHE_PATH.exists():
    with np.load(CACHE_PATH) as cache:
        cache_matches = (
            cache["model_revision"].item()
            == MODEL_REVISION
            and cache["corpus_sha256"].item()
            == corpus_sha256
            and cache["passage_max_length"].item()
            == PASSAGE_MAX_LENGTH
            and cache["section_ids"].tolist()
            == section_ids
        )

        if cache_matches:
            passage_embeddings = (
                torch.from_numpy(
                    cache["embeddings"]
                ).cuda()
            )
            passage_cache_hit = True
            print(
                "Loaded passage embeddings from "
                f"{CACHE_PATH}"
            )
        else:
            print(
                "Existing passage cache does not "
                "match; rebuilding"
            )

if passage_embeddings is None:
    print("Encoding all passage sections")

    passage_embeddings = encode_texts(
        model=model,
        texts=[
            section["text"]
            for section in sections
        ],
        instruction="",
        max_length=PASSAGE_MAX_LENGTH,
        batch_size=PASSAGE_BATCH_SIZE,
    )

    CACHE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    np.savez_compressed(
        CACHE_PATH,
        embeddings=(
            passage_embeddings.cpu().numpy()
        ),
        section_ids=np.asarray(section_ids),
        model_revision=np.asarray(
            MODEL_REVISION
        ),
        corpus_sha256=np.asarray(
            corpus_sha256
        ),
        passage_max_length=np.asarray(
            PASSAGE_MAX_LENGTH
        ),
    )
    passage_embeddings = (
        passage_embeddings.cuda()
    )

    print(
        "Cached passage embeddings at "
        f"{CACHE_PATH}"
    )

print(
    "Encoding all evaluation queries "
    f"with batch size {QUERY_BATCH_SIZE}"
)

query_encode_start = time.perf_counter()

query_embeddings = encode_texts(
    model=model,
    texts=[
        query["query"]
        for query in queries
    ],
    instruction=QUERY_INSTRUCTION,
    max_length=QUERY_MAX_LENGTH,
    batch_size=QUERY_BATCH_SIZE,
)

query_encode_seconds = (
    time.perf_counter() - query_encode_start
)

print(
    "Query encoding seconds: "
    f"{query_encode_seconds:.2f}"
)

ranking_start = time.perf_counter()
scores = query_embeddings @ passage_embeddings.T

ranks = []
ranks_by_leaf = defaultdict(list)
query_results = []

for query_number, query in enumerate(queries):
    score_values = (
        scores[query_number]
        .detach()
        .cpu()
        .tolist()
    )

    ranked_indices = sorted(
        range(len(sections)),
        key=lambda index: (
            -score_values[index],
            sections[index]["id"],
        ),
    )

    gold_index = section_index[
        query["gold_leaf_id"]
    ]
    gold_rank = (
        ranked_indices.index(gold_index) + 1
    )

    ranks.append(gold_rank)
    ranks_by_leaf[
        query["gold_leaf_id"]
    ].append(gold_rank)

    top_predictions = []

    for rank, candidate_index in enumerate(
        ranked_indices[:5],
        start=1,
    ):
        section = sections[candidate_index]
        top_predictions.append(
            {
                "rank": rank,
                "leaf_id": section["id"],
                "title": section["title"],
                "path": section["path"],
                "score": (
                    score_values[
                        candidate_index
                    ]
                    * 100
                ),
                "is_gold": (
                    section["id"]
                    == query["gold_leaf_id"]
                ),
            }
        )

    query_results.append(
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
            "top_predictions": (
                top_predictions
            ),
        }
    )

ranking_seconds = (
    time.perf_counter() - ranking_start
)

micro_metrics = metrics_from_ranks(ranks)

per_leaf_metrics = {
    leaf_id: metrics_from_ranks(leaf_ranks)
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

peak_gpu_gib = (
    torch.cuda.max_memory_allocated()
    / 1024**3
)

payload = {
    "retriever": "NV-Embed-v2",
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "query_instruction": QUERY_INSTRUCTION,
    "passage_max_length": (
        PASSAGE_MAX_LENGTH
    ),
    "query_max_length": QUERY_MAX_LENGTH,
    "query_batch_size": QUERY_BATCH_SIZE,
    "sections_path": str(SECTIONS_PATH),
    "corpus_sha256": corpus_sha256,
    "queries_path": str(QUERIES_PATH),
    "queries_sha256": queries_sha256,
    "section_count": len(sections),
    "query_count": len(queries),
    "evaluated_leaf_count": len(
        ranks_by_leaf
    ),
    "passage_cache_hit": passage_cache_hit,
    "model_load_seconds": (
        model_load_seconds
    ),
    "query_encode_seconds": (
        query_encode_seconds
    ),
    "ranking_seconds": ranking_seconds,
    "milliseconds_per_query_encoding": (
        query_encode_seconds
        / len(queries)
        * 1000
    ),
    "peak_gpu_memory_gib": peak_gpu_gib,
    "micro_metrics": micro_metrics,
    "macro_leaf_metrics": macro_metrics,
    "per_leaf_metrics": per_leaf_metrics,
    "results": query_results,
}

RESULTS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)
RESULTS_PATH.write_text(
    json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)

print("\nNV-Embed evaluation complete")
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
    "Query encoding ms/query: "
    f"{payload['milliseconds_per_query_encoding']:.3f}"
)
print(
    "Peak GPU memory: "
    f"{peak_gpu_gib:.2f} GiB"
)
print(f"Wrote: {RESULTS_PATH}")
