from collections import defaultdict
from pathlib import Path
import argparse
import json
import math
import time

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_retriever_lora import (
    DATA_DIR,
    DEFAULTS,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SNAPSHOT,
    SEED,
    build_target_catalog,
    build_token_trie,
    read_jsonl,
    render_prompt,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=("dsi", "stair"),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--limit",
        type=int,
    )
    parser.add_argument(
        "--output",
        type=Path,
    )
    return parser.parse_args()


def evaluate(
    model,
    tokenizer,
    records,
    template,
    toc,
    target_to_leaf,
    batch_size,
    top_k,
):
    candidate_targets = list(target_to_leaf)

    if top_k > len(candidate_targets):
        raise ValueError(
            f"top-k {top_k} exceeds "
            f"{len(candidate_targets)} candidates"
        )

    trie = build_token_trie(
        tokenizer,
        candidate_targets,
    )

    tokenizer.padding_side = "left"
    model.eval()
    model.config.use_cache = True

    predictions = []
    leaf_totals = defaultdict(int)
    leaf_correct_at_1 = defaultdict(int)

    start = time.perf_counter()

    for offset in range(0, len(records), batch_size):
        batch_records = records[offset : offset + batch_size]
        prompt_texts = [
            render_prompt(
                tokenizer,
                template,
                toc,
                record["query"],
            )
            for record in batch_records
        ]

        inputs = tokenizer(
            prompt_texts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        input_width = inputs["input_ids"].shape[1]

        def allowed_tokens(batch_id, input_ids):
            del batch_id
            generated = input_ids[input_width:].tolist()
            node = trie

            for token_id in generated:
                if token_id not in node:
                    return [tokenizer.eos_token_id]
                node = node[token_id]

            allowed = list(node)
            return allowed or [tokenizer.eos_token_id]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                num_beams=top_k,
                num_return_sequences=top_k,
                return_dict_in_generate=True,
                output_scores=True,
                early_stopping=True,
                prefix_allowed_tokens_fn=allowed_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = outputs.sequences[:, input_width:]
        decoded = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )
        sequence_scores = (
            outputs.sequences_scores
            .detach()
            .float()
            .cpu()
            .tolist()
        )

        expected_sequences = len(batch_records) * top_k

        if len(decoded) != expected_sequences:
            raise ValueError(
                "Unexpected beam count: "
                f"{len(decoded)} != {expected_sequences}"
            )

        for local_index, record in enumerate(batch_records):
            start_index = local_index * top_k
            end_index = start_index + top_k
            targets = [
                text.strip()
                for text in decoded[start_index:end_index]
            ]
            scores = sequence_scores[start_index:end_index]

            unknown = [
                target
                for target in targets
                if target not in target_to_leaf
            ]

            if unknown:
                raise ValueError(
                    "Constrained decoding produced unknown "
                    f"targets: {unknown}"
                )

            ranked_leaf_ids = [
                target_to_leaf[target]
                for target in targets
            ]
            gold_leaf_id = record["gold_leaf_id"]
            rank = next(
                (
                    index
                    for index, leaf_id in enumerate(
                        ranked_leaf_ids,
                        start=1,
                    )
                    if leaf_id == gold_leaf_id
                ),
                None,
            )

            leaf_totals[gold_leaf_id] += 1
            leaf_correct_at_1[gold_leaf_id] += int(rank == 1)

            predictions.append(
                {
                    "id": record["id"],
                    "query": record["query"],
                    "gold_leaf_id": gold_leaf_id,
                    "source_unit_id": record["source_unit_id"],
                    "gold_rank_at_k": rank,
                    "ranked_targets": [
                        {
                            "rank": index,
                            "target": target,
                            "leaf_id": leaf_id,
                            "sequence_score": score,
                        }
                        for index, (target, leaf_id, score) in enumerate(
                            zip(targets, ranked_leaf_ids, scores),
                            start=1,
                        )
                    ],
                }
            )

        completed = offset + len(batch_records)
        print(
            f"Evaluated {completed}/{len(records)} queries",
            flush=True,
        )

    elapsed = time.perf_counter() - start
    ranks = [
        prediction["gold_rank_at_k"]
        for prediction in predictions
    ]
    total = len(ranks)

    def recall_at(k):
        return sum(
            rank is not None and rank <= k
            for rank in ranks
        ) / total

    macro_leaf_recall_at_1 = float(
        np.mean(
            [
                leaf_correct_at_1[leaf_id]
                / leaf_totals[leaf_id]
                for leaf_id in sorted(leaf_totals)
            ]
        )
    )

    summary = {
        "queries": total,
        "candidates": len(candidate_targets),
        "evaluated_leaves": len(leaf_totals),
        "recall_at_1": recall_at(1),
        "recall_at_3": recall_at(min(3, top_k)),
        "recall_at_5": recall_at(min(5, top_k)),
        "mrr_at_5": sum(
            1.0 / rank if rank is not None else 0.0
            for rank in ranks
        ) / total,
        "ndcg_at_3": sum(
            (
                1.0 / math.log2(rank + 1)
                if rank is not None and rank <= 3
                else 0.0
            )
            for rank in ranks
        ) / total,
        "macro_leaf_recall_at_1": macro_leaf_recall_at_1,
        "seconds": elapsed,
        "milliseconds_per_query": elapsed * 1000 / total,
        "peak_gpu_gib": (
            torch.cuda.max_memory_allocated()
            / 1024**3
        ),
    }

    return summary, predictions


def main():
    args = parse_args()
    set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()

    manifest = json.loads(
        (DATA_DIR / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    labels = json.loads(
        (DATA_DIR / "labels.json").read_text(
            encoding="utf-8"
        )
    )
    toc = (DATA_DIR / "toc.txt").read_text(
        encoding="utf-8"
    ).rstrip()
    records = read_jsonl(
        DATA_DIR / args.method / "test.jsonl"
    )

    if args.limit is not None:
        records = records[: args.limit]

    template = manifest["prompt_templates"][args.method]
    target_to_leaf = build_target_catalog(
        args.method,
        labels,
    )
    batch_size = args.batch_size or (
        16 if args.method == "dsi" else 4
    )
    adapter_path = (
        Path("artifacts/models")
        / args.method
        / "best-adapter"
    )

    if not adapter_path.exists():
        raise FileNotFoundError(adapter_path)

    output_path = args.output

    if output_path is None:
        suffix = "-smoke" if args.limit is not None else ""
        output_path = Path(
            "artifacts/results/"
            f"whole-child-{args.method}-test{suffix}.json"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_SNAPSHOT,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(
        f"Loading {MODEL_ID} and {adapter_path}",
        flush=True,
    )
    load_start = time.perf_counter()
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_SNAPSHOT,
        local_files_only=True,
        use_safetensors=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=False,
    )
    model.eval()
    print(
        "Model and adapter load seconds: "
        f"{time.perf_counter() - load_start:.2f}",
        flush=True,
    )

    summary, predictions = evaluate(
        model=model,
        tokenizer=tokenizer,
        records=records,
        template=template,
        toc=toc,
        target_to_leaf=target_to_leaf,
        batch_size=batch_size,
        top_k=args.top_k,
    )

    payload = {
        "method": args.method,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "adapter_path": str(adapter_path),
        "split": "test",
        "seed": SEED,
        "top_k": args.top_k,
        "batch_size": batch_size,
        "limited": args.limit is not None,
        "summary": summary,
        "predictions": predictions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n{args.method.upper()} test evaluation complete")
    print(f"Queries: {summary['queries']}")
    print(f"Candidates: {summary['candidates']}")
    print(f"Evaluated leaves: {summary['evaluated_leaves']}")
    print(f"Recall@1: {summary['recall_at_1']:.4f}")
    print(f"Recall@3: {summary['recall_at_3']:.4f}")
    print(f"Recall@5: {summary['recall_at_5']:.4f}")
    print(f"MRR@5: {summary['mrr_at_5']:.4f}")
    print(f"nDCG@3: {summary['ndcg_at_3']:.4f}")
    print(
        "Macro leaf Recall@1: "
        f"{summary['macro_leaf_recall_at_1']:.4f}"
    )
    print(
        "Milliseconds/query: "
        f"{summary['milliseconds_per_query']:.3f}"
    )
    print(f"Peak GPU memory: {summary['peak_gpu_gib']:.2f} GiB")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
