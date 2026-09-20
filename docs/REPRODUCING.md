# Reproducing the Whole Child experiment

This guide has three levels. Choose the smallest one that answers your
question:

1. **Release validation** checks the tracked data, notebook, and frozen
   metrics with the Python standard library.
2. **CPU reproduction** re-runs BM25 and regenerates the figures.
3. **Full reproduction** rebuilds the corpus, generates supervision, encodes
   the dense baseline, trains two LoRA adapters, and evaluates all systems.

The full path is intentionally staged. The vLLM, NV-Embed, and retriever
training stacks use separate virtual environments because their Transformers,
PyTorch, CUDA, and tokenizer requirements are not interchangeable.

## Known-good platform

- Amazon SageMaker Studio JupyterLab
- Python 3.12
- Linux x86-64
- NVIDIA L40S, 46,060 MiB visible GPU memory
- CUDA-enabled PyTorch 2.8.0 supplied by SageMaker for NV-Embed and training
- 100 GB persistent volume

Allow at least 80 GB of free disk if all three model snapshots will coexist.
Cold-loading large safetensor shards from a reattached volume can take much
longer than inference. In this run, NV-Embed's 15.7 GB checkpoint took 45.5
minutes to cold-load, while encoding all 1,758 test queries took 13.25 seconds.

## Immutable model inputs

| Role | Model | Revision |
|---|---|---|
| Question generation | `casperhansen/mixtral-instruct-awq` | `0a898130957afe22021bbaf807f50f6bbce88201` |
| Dense retrieval | `nvidia/NV-Embed-v2` | `3fa59658547db50a1e8e3346cf057fd0c77ed6ef` |
| DSI and STAIR | `mistralai/Mistral-7B-Instruct-v0.2` | `63a8b081895390a26e140280378bc85ec8bce07a` |

All scripts load local snapshots at those revisions. Model weights are not
stored in Git and retain their upstream licenses.

## Level 1: validate the release

From the repository root:

```bash
python3 scripts/validate_repository.py
python3 -m compileall -q scripts
```

This does not need the raw PDF, ignored generated-question records, model
weights, or GPU libraries. It validates the tracked JSON/JSONL files, frozen
counts and metrics, split leakage constraints, retriever projections, and
notebook syntax/IDs.

## Level 2: CPU baseline and report

```bash
make setup-core
make evaluate-bm25
make report
```

The BM25 evaluator reads the tracked leaf sections and frozen test split. It
writes its detailed output under ignored `artifacts/results/`. The plotter
reads the versioned `results/whole-child-results.json` and rebuilds the five
PNG files in `reports/figures/`.

To regenerate the clean notebook source:

```bash
make notebook
```

To execute it, install Jupyter/nbconvert in an environment of your choice and
run:

```bash
make execute-notebook JUPYTER=/path/to/jupyter
```

## Level 3: full reproduction

### 1. Obtain and verify the source PDF

Download *The Whole Child: Development in the Early Years* from the
[ROTEL Pressbooks site](https://rotel.pressbooks.pub/whole-child/) and place
the PDF at:

```text
data/raw/whole-child.pdf
```

Verify the exact artifact used here:

```bash
sha256sum data/raw/whole-child.pdf
```

Expected:

```text
14c95ceb029cafaf98f9f2c67e3b73baa8177b53ca3198519c3a2ec19c547d75
```

The raw PDF is ignored because it is third-party CC BY-NC-SA 4.0 material.

### 2. Create the CPU preprocessing environment

```bash
make setup-core
make derive-corpus
```

The derivation sequence:

1. reads the native PDF ToC and builds stable node/leaf identifiers;
2. locates headings on their declared pages;
3. extracts 129 leaf sections;
4. reconstructs block-aware paragraphs;
5. removes explicit layout/attribution noise; and
6. merges fragments into semantic generation units while retaining source
   paragraph provenance.

The expected endpoint is 181 ToC nodes, 129 retrieval leaves, 1,019 raw
paragraph blocks, 115 removed noise blocks, and 555 generation units.

### 3. Create the local Mixtral/vLLM environment

```bash
make setup-mixtral
source .venv-mixtral/bin/activate

hf download casperhansen/mixtral-instruct-awq \
  --revision 0a898130957afe22021bbaf807f50f6bbce88201
```

The explicit cu129 vLLM wheel in `requirements-mixtral.txt` avoids the CUDA 13
binary that failed on the known-good SageMaker image. The generation script
also disables the optional FlashInfer sampler before importing vLLM because
that sampler's JIT warmup required CUDA development headers not present on the
image. Model inference still uses the supported native sampler.

Plan the run without loading the model:

```bash
python scripts/generate_questions.py --plan
```

Generate with resumable JSONL checkpoints:

```bash
python scripts/generate_questions.py --batch-size 8
```

Generated records and quarantined raw attempts live under
`artifacts/generated/`, which is ignored. The validator expects 555 valid
units, 3,214 questions, no missing expected units, and no exact global
duplicate questions. Eleven failed attempts remain as audit history; one
source unit uses six deterministic list-grounded repair questions restricted
to training.

Return to the CPU environment and build the grouped split:

```bash
deactivate
source .venv/bin/activate
make validate-questions
make split-data
```

Expected split counts:

| Split | Questions | Source units | Leaf coverage |
|---|---:|---:|---:|
| Train | 1,064 | 171 | 129 |
| Development | 392 | 88 | 66 |
| Test | 1,758 | 296 | 86 |

The source unit is the indivisible assignment group; related questions never
cross split boundaries.

### 4. Run BM25

```bash
make evaluate-bm25
```

Expected Recall@1 is 0.7986 on the frozen test split.

### 5. Create the NV-Embed environment and evaluate

The known-good SageMaker image already supplied CUDA-enabled PyTorch 2.8.0.
The venv inherits it while pinning the model-specific Transformers stack:

```bash
make setup-nvembed
source .venv-nvembed/bin/activate

hf download nvidia/NV-Embed-v2 \
  --revision 3fa59658547db50a1e8e3346cf057fd0c77ed6ef

python scripts/evaluate_nvembed.py
```

The first run writes a passage-embedding cache under
`artifacts/embeddings/`; later runs validate its corpus hash before reuse.
Expected Recall@1 is 0.6553.

### 6. Create the retriever environment and data

The known-good SageMaker base environment supplied PyTorch 2.8.0,
Transformers 4.57.6, and Accelerate 1.14.0. The dedicated environment adds
the pinned dataset and PEFT packages:

```bash
deactivate
make setup-retriever
source .venv-retriever/bin/activate

hf download mistralai/Mistral-7B-Instruct-v0.2 \
  --revision 63a8b081895390a26e140280378bc85ec8bce07a

make prepare-retriever
```

The prepared records project the same questions into two tasks:

- DSI: `query -> whole-child-NNN`
- STAIR: `complete ToC + query -> Chapter > Section > Leaf`

The ToC is 1,603 Mistral tokens; the largest STAIR prompt is 1,687 tokens and
the largest target path is 39 tokens.

### 7. Smoke-test training

```bash
python scripts/smoke_retriever_lora.py
python scripts/train_retriever_lora.py --method dsi --smoke --fresh
python scripts/train_retriever_lora.py --method stair --smoke --fresh
```

This validates forward/backward passes, masking, LoRA insertion, constrained
decoding, checkpoint selection, and resume state before the full jobs.

### 8. Train DSI and STAIR

```bash
make train-dsi
make train-stair
```

Shared settings:

- Mistral-7B-Instruct-v0.2 in BF16
- LoRA rank 16, alpha 32, dropout 0.05
- attention and MLP projection targets
- effective batch size 16
- learning rate 2e-4 with a linear schedule
- seed 42
- five-epoch ceiling and best checkpoint selected by development Recall@1

DSI uses batch size 8 with two accumulation steps and a maximum length of
512. STAIR uses batch size 1 with 16 accumulation steps and a maximum length
of 2,048. Both use trie-constrained valid-target decoding during evaluation.

The selected DSI adapter is epoch 4 (`dev R@1 = 0.3750`). STAIR improves
through epoch 5 (`dev R@1 = 0.4923`), so its five-epoch result should be read
as compute-bounded rather than converged.

### 9. Evaluate and compare

```bash
make evaluate-dsi
make evaluate-stair
make compare
```

The evaluators use five constrained beams and retain per-query rankings under
`artifacts/results/`. The paired analysis computes exact McNemar, a seeded
10,000-sample paired bootstrap, depth strata, development-label coverage,
repeated-title behavior, and structural error categories.

Expected rank-1 results:

```text
DSI   0.4010
STAIR 0.3754
```

### 10. Freeze and render the release

```bash
make freeze-results
make report
make notebook
python3 scripts/validate_repository.py
```

`freeze_results.py` verifies the detailed artifacts and writes the compact
versioned result files. Because the detailed predictions and adapters are
ignored, the frozen JSON also records their paths, byte sizes, and SHA-256
hashes.

## Resume and artifact behavior

- Question generation appends one JSONL record per completed unit and skips
  already-valid units on restart.
- Retriever training stores `state.json`, `training-state.pt`, a last adapter,
  and a best adapter. Omit `--fresh` to resume a compatible interrupted run.
- `--fresh` deliberately starts a new run and should only be used when prior
  output is no longer needed or has been backed up.
- Everything below `artifacts/` is ignored by Git. Copy expensive outputs to
  durable object storage before deleting a cloud workspace.

## Interpreting differences

A numerically different reproduction is not automatically a bug. Check, in
order:

1. PDF SHA and page count;
2. retrieval-leaf count and canonical paths;
3. generation-unit and question counts;
4. grouped-split manifest hashes;
5. immutable model revisions;
6. best development checkpoint, not last checkpoint;
7. constrained-decoding candidate catalog; and
8. exact test query/candidate counts.

The original paper used 18 books and an unavailable data/code release. This
repository's result is therefore an independently reconstructed one-book
stress test, not an exact rerun of SearchTome.
