# Whole Child retrieval results

These are the frozen results from an independent, one-book reproduction of
STAIR. All four retrievers use the same leakage-safe 1,758-query test split,
129 candidate leaves, and 86 represented gold leaves.

| Retriever | Recall@1 | Recall@3 | Recall@5 | nDCG@3 | Macro leaf Recall@1 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.7986 | 0.9107 | 0.9357 | 0.8648 | 0.7999 |
| NV-Embed-v2 | 0.6553 | 0.8271 | 0.8697 | 0.7553 | 0.7535 |
| DSI | 0.4010 | 0.5228 | 0.5830 | 0.4716 | 0.3957 |
| STAIR | 0.3754 | 0.4812 | 0.4994 | 0.4373 | 0.3295 |

BM25 and NV-Embed report full-candidate MRR. DSI and STAIR report MRR@5
because their constrained decoder retains five ranked beams.

## DSI versus STAIR

DSI reaches 0.4010 Recall@1 and STAIR reaches
0.3754, a STAIR-minus-DSI difference of
-0.0256. The exact paired McNemar p-value is
0.054252. The paired-bootstrap 95%
interval is [-0.0506,
+0.0000].

The aggregate result hides a strong depth interaction. STAIR improves
Recall@1 by 0.2644 on depth-2 queries but trails DSI by 0.1061 on depth-3
queries. This independent reproduction therefore does not reproduce the
paper's aggregate STAIR-over-DSI ordering, while still finding substantial
benefit from explicit structure for coarse routing.

Machine-readable metrics, training state, and SHA-256 artifact checksums are
stored in `results/whole-child-results.json`.
