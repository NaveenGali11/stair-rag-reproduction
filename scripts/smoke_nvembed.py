import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel


MODEL_ID = "nvidia/NV-Embed-v2"
MODEL_REVISION = "3fa59658547db50a1e8e3346cf057fd0c77ed6ef"

SECTIONS_PATH = Path("data/derived/whole-child-sections.json")
QUERIES_PATH = Path("data/gold_queries.jsonl")

# BM25 ranked this leaf second, making it a useful initial distractor.
DISTRACTOR_ID = "whole-child-155"

QUERY_INSTRUCTION = (
    "Instruct: Given a question, retrieve passages that answer the question\n"
    "Query: "
)


def gibibytes(byte_count: int) -> float:
    return byte_count / (1024**3)


with QUERIES_PATH.open(encoding="utf-8") as query_file:
    gold_query = json.loads(next(line for line in query_file if line.strip()))

sections = json.loads(SECTIONS_PATH.read_text(encoding="utf-8"))
sections_by_id = {section["id"]: section for section in sections}

gold_id = gold_query["gold_leaf_id"]
candidate_ids = [gold_id, DISTRACTOR_ID]
candidates = [sections_by_id[section_id] for section_id in candidate_ids]

print(f"Loading {MODEL_ID} at revision {MODEL_REVISION}")
print(f"Query: {gold_query['query']}")
print(f"Gold leaf: {gold_id}")

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
    "GPU memory allocated after loading: "
    f"{gibibytes(torch.cuda.memory_allocated()):.2f} GiB"
)

passages = [candidate["text"] for candidate in candidates]

torch.cuda.reset_peak_memory_stats()

with torch.inference_mode():
    query_embedding = model.encode(
        [gold_query["query"]],
        instruction=QUERY_INSTRUCTION,
        max_length=256,
    )
    passage_embeddings = model.encode(
        passages,
        instruction="",
        max_length=512,
    )

    query_embedding = F.normalize(query_embedding, p=2, dim=1)
    passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1)
    scores = (query_embedding @ passage_embeddings.T).squeeze(0) * 100

print(f"Query embedding shape: {tuple(query_embedding.shape)}")
print(f"Passage embedding shape: {tuple(passage_embeddings.shape)}")
print(
    "Peak GPU memory during encoding: "
    f"{gibibytes(torch.cuda.max_memory_allocated()):.2f} GiB"
)

ranked_indices = torch.argsort(scores, descending=True).cpu().tolist()

print("\nRanking:")
for rank, candidate_index in enumerate(ranked_indices, start=1):
    candidate = candidates[candidate_index]
    candidate_id = candidate["id"]
    marker = " <-- GOLD" if candidate_id == gold_id else ""
    path = " > ".join(candidate["path"])

    print(
        f"{rank}. {candidate_id} | "
        f"{scores[candidate_index].item():.3f} | "
        f"{path}{marker}"
    )