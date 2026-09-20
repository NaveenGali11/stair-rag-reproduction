# Teaching an LLM to navigate a book

## What an independent STAIR reproduction taught us about document structure

Most retrieval systems flatten documents before they search them. A textbook
becomes windows of tokens; a manual becomes chunks; a report becomes vectors.
The document's own organization—chapters, sections, and subsections—is often
reduced to metadata or discarded.

The 2026 STAIR paper asks a compelling question: what if that hierarchy is
part of the retriever itself? STAIR gives a fine-tuned language model a
document's complete Table of Contents (ToC) and asks it to generate the leaf
section most likely to answer a query. The model is not searching independent
passages. It is navigating a semantic map.

We independently reconstructed that idea on one open textbook, compared it
with BM25, dense retrieval, and a DSI control, and found a result more useful
than a clean win: STAIR helped dramatically at one hierarchy depth and hurt at
another.

## Four ways to retrieve a section

The experiment compares four retrieval families over the same 129 textbook
leaf sections.

**BM25** builds a lexical index. It rewards overlap between query terms and
section text.

```text
query -> token overlap -> ranked sections
```

**NV-Embed-v2** encodes queries and sections into dense vectors and ranks by
similarity.

```text
query -> embedding -> nearest section embeddings
```

**DSI** stores the retrieval mapping in a fine-tuned language model. Given a
query, it generates an opaque section identifier.

```text
query -> fine-tuned Mistral -> whole-child-023
```

**STAIR** is also generative retrieval, but it exposes the complete hierarchy
and generates the full canonical path.

```text
ToC + query -> fine-tuned Mistral -> Chapter > Section > Leaf
```

That last distinction is important. Our end-to-end comparison changes both
the input and the target: STAIR sees the ToC and must generate a longer path,
while DSI sees only the query and emits a short ID. This follows the systems
described in the paper, but it does not isolate the ToC's effect from the
difficulty of path generation. We return to that limitation later.

## The paper's claim

Kumar et al. introduce SearchTome, a benchmark built from 18 books across six
domains. Their reported average Recall@1 ordering is:

```text
STAIR 82.6 > DSI 76.9 > DPR 68.7 > BM25 59.5
```

They report that STAIR's improvement over DSI is statistically significant.
The original anonymous code/data link was no longer available when we began,
so our work is an independent reconstruction from the paper and public source
material—not a byte-for-byte rerun.

We selected *The Whole Child: Development in the Early Years*, one of the
books listed in SearchTome. The paper reports 129 leaves and 3,190 questions
for this title. That made it small enough to audit manually while still
containing a real three-level hierarchy.

## Reconstructing a corpus is part of the experiment

The current publisher PDF is 183 pages; the paper reports 182. We fingerprinted
the PDF, extracted all 181 native ToC nodes, and applied one semantic rule: a
metadata-only *Media Attributions* leaf is not a retrieval destination. That
left exactly 129 content leaves.

PDF extraction was not a simple call to `get_text()`. Running footers looked
like headings. Diagram labels duplicated headings. Visual line wraps looked
like paragraph boundaries. On one page, a diagram's “Medulla oblongata” label
appeared before the actual section heading and initially captured the wrong
boundary.

The final pipeline uses block geometry, heading priority, left-margin
alignment, stable ToC IDs, and regression checks for ambiguous cases. It
produced 1,019 provenance-carrying paragraph blocks across all 129 leaves.

We then built a separate cleaning layer rather than mutating extraction
heuristics until the text looked right. It removed attribution tails, naked
URLs, and two Pressbooks placeholders for interactive elements that were not
present in the PDF. It also merged labels, bullets, and layout fragments into
555 semantic question-generation units. Every merged unit retains the ordered
IDs of its source blocks.

This separation matters: raw extraction errors and downstream modeling choices
remain independently inspectable.

## Synthetic supervision without split leakage

We used a pinned 4-bit AWQ build of Mixtral 8x7B Instruct to generate questions
from each semantic unit. Passage text was the only semantic input; ToC labels
were not shown to the generator.

Generation was resumable and deliberately strict. Every completed unit was
checkpointed to JSONL. Invalid arrays, nested outputs, non-string values, and
truncated responses were quarantined rather than silently accepted. Eleven
failed attempts remain as audit history. One stubborn six-item teaching list
was repaired with six deterministic, passage-grounded questions, marked as
repair data and restricted to training.

The validated result contains 3,214 questions across 555 units and all 129
leaves, with no unresolved units or exact duplicate questions.

The unit—not the question—is the split boundary. All questions derived from a
passage remain together, preventing near-duplicate supervision from leaking
between train and test. The resulting split contains:

| Split | Questions | Source units | Leaf coverage |
|---|---:|---:|---:|
| Train | 1,064 | 171 | 129 |
| Development | 392 | 88 | 66 |
| Test | 1,758 | 296 | 86 |

Forty-three leaves contain only one independent generation unit, so under this
policy they must remain train-only. This later turns out to matter.

## A qualitative error reproduced before training

The paper highlights this query:

> How do preschoolers react when caregivers and teachers belittle their
> autonomous actions?

The correct section is *Initiative vs. Guilt (Preschool Years)*. NV-Embed-v2
ranked *Autonomy vs. Shame/Doubt (Toddlerhood)* first and the correct section
second—the same hard-negative behavior described in the paper. “Autonomous
actions” pulls the dense model toward the autonomy section, while “preschoolers”
and the consequence of adult discouragement identify initiative versus guilt.

Our `rank-bm25` implementation ranked the correct section first, unlike the
paper's Elasticsearch BM25 result. We treat that as an implementation
difference, not as a successful reproduction of the paper's BM25 row.

## Training DSI and STAIR under one budget

Both generative retrievers use the same immutable
Mistral-7B-Instruct-v0.2 revision, training questions, seed, BF16 precision,
LoRA rank 16, LoRA alpha 32, effective batch size 16, learning-rate schedule,
candidate set, and five-epoch ceiling. Evaluation uses trie-constrained valid
targets and five beams.

DSI's best development checkpoint occurs at epoch 4 with 0.3750 Recall@1.
STAIR improves in every measured epoch and reaches 0.4923 at epoch 5. That is
an important caveat: the STAIR run ends because the shared budget ends, not
because its development metric has converged.

## The result

On the untouched 1,758-query test set:

| Retriever | Recall@1 | Recall@3 | Recall@5 | nDCG@3 |
|---|---:|---:|---:|---:|
| BM25 | **0.7986** | **0.9107** | **0.9357** | **0.8648** |
| NV-Embed-v2 | 0.6553 | 0.8271 | 0.8697 | 0.7553 |
| DSI | 0.4010 | 0.5228 | 0.5830 | 0.4716 |
| STAIR | 0.3754 | 0.4812 | 0.4994 | 0.4373 |

STAIR does not beat DSI overall. DSI leads by 2.56 Recall@1 points.

The paired result is close rather than decisive. Both systems are correct on
421 questions, only DSI is correct on 284, only STAIR on 239, and both fail on
814. Exact two-sided McNemar gives `p = 0.0543`. A 10,000-sample paired
bootstrap puts the 95% interval for STAIR minus DSI at `[-0.0506, 0.0000]`.

So the point estimate favors DSI, but it narrowly misses the conventional
0.05 significance threshold.

## The aggregate hides two different experiments

Hierarchy depth reverses the conclusion.

On 382 depth-2 queries, STAIR reaches 0.5497 Recall@1 versus DSI's 0.2853—a
gain of 26.44 points and about 101 additional correct answers.

On 1,376 depth-3 queries, STAIR falls to 0.3270 while DSI reaches 0.4331—a
loss of 10.61 points and about 146 answers. Because depth-3 queries dominate
the test set, they erase STAIR's large coarse-routing advantage.

Development-label coverage exposes another fault line. On 1,614 queries whose
gold leaves also occur in development, the models are close: 0.4002 for DSI
and 0.3916 for STAIR. On the remaining 144 queries, DSI scores 0.4097 while
STAIR drops to 0.1944. Those queries are only 8.2% of the test set but explain
roughly 31 of DSI's 45 additional correct answers.

Our interpretation is narrower than “structure works” or “structure fails”:
under this data and training budget, a complete ToC is a powerful coarse
router, but generating exact deeper paths remains difficult—especially when
checkpoint selection never observes those labels in development.

## Are STAIR's mistakes at least structurally closer?

Only slightly.

Among DSI's incorrect depth-3 predictions, 32.7% remain inside the correct
chapter. For STAIR, 34.0% do. STAIR makes more cross-chapter errors in absolute
terms: 611 versus 525.

On 304 queries whose leaf title appears elsewhere in the book, DSI reaches
0.5789 Recall@1 and STAIR reaches 0.5559. STAIR's misses are somewhat more
likely to remain in the correct chapter, but that locality does not translate
into better exact retrieval.

## Operationally, the hierarchy is expensive

The complete ToC is 1,603 Mistral tokens before the query and chat framing.
Repeating it for every request increases prefill, memory, and beam-search cost.

On the same L40S, DSI averaged 69.4 ms/query and peaked at 16.24 GiB during
evaluation. STAIR averaged 2.17 seconds/query and peaked at 36.92 GiB. Those
numbers are implementation- and hardware-specific, but the direction is not
surprising: global structure is useful context, not free context.

Storage behavior was another practical lesson. A cold NV-Embed checkpoint load
from the reattached volume took 45.5 minutes; query encoding took 13.25
seconds. A cold Mixtral load took almost 32 minutes, while a warm-cache restart
took 6.83 seconds. Model evaluation can easily become a storage benchmark if
cold-start time is not separated from inference time.

## What we can and cannot conclude

We cannot conclude that the paper's multi-book result is wrong. Our source PDF
is a newer page version, the original release and generation prompt were
unavailable, our split is passage-grouped, we use one book and one seed, and
the STAIR development curve is still improving at the shared budget limit.

We can conclude that the aggregate STAIR advantage is not automatic. It
depends on hierarchy depth, target representation, development coverage,
training budget, and evaluation design.

The clean next experiment is a two-by-two ablation:

| | Opaque ID target | Canonical path target |
|---|---|---|
| Query only | DSI-like | path-generation control |
| ToC + query | ToC-input control | STAIR |

That design would separate the benefit of seeing the hierarchy from the cost
of generating it.

## The main lesson

Document structure should not be treated as decorative metadata. In this
experiment, it changed what the model could retrieve—but not uniformly.

STAIR was genuinely strong at broad navigation and materially weak at deeper
exact routing. That split result is more informative than a single leaderboard
number. It tells us where to invest next: hierarchical objectives, depth-aware
training, better development coverage, cached/prefix-efficient ToC handling,
and an ablation that disentangles input structure from output representation.

The complete code, tracked data, executed notebook, frozen metrics, and
reproduction instructions are available in the
[repository](https://github.com/NaveenGali11/stair-rag-reproduction). The
original paper is [arXiv:2609.03874](https://arxiv.org/abs/2609.03874).
