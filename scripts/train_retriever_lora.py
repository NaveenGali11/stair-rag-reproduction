from pathlib import Path
import argparse
import json
import math
import random
import time

import numpy as np
import torch
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
)
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = (
    "63a8b081895390a26e140280378bc85ec8bce07a"
)
MODEL_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/"
    "models--mistralai--Mistral-7B-Instruct-v0.2/"
    "snapshots/"
    / MODEL_REVISION
)
DATA_DIR = Path("data/retriever/whole-child")

SEED = 42
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

DEFAULTS = {
    "dsi": {
        "batch_size": 8,
        "gradient_accumulation": 2,
        "eval_batch_size": 32,
    },
    "stair": {
        "batch_size": 1,
        "gradient_accumulation": 16,
        "eval_batch_size": 8,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=("dsi", "stair"),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def render_prompt(
    tokenizer,
    template,
    toc,
    query,
):
    prompt = template.format(
        query=query,
        toc=toc,
    )

    return tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


class RetrieverDataset(Dataset):
    def __init__(
        self,
        records,
        tokenizer,
        template,
        toc,
        maximum_length,
    ):
        self.examples = []

        for record in records:
            prompt_text = render_prompt(
                tokenizer,
                template,
                toc,
                record["query"],
            )
            prompt_ids = tokenizer(
                prompt_text,
                add_special_tokens=False,
            )["input_ids"]
            target_ids = tokenizer(
                record["target"],
                add_special_tokens=False,
            )["input_ids"]

            input_ids = (
                prompt_ids
                + target_ids
                + [tokenizer.eos_token_id]
            )
            labels = (
                [-100] * len(prompt_ids)
                + target_ids
                + [tokenizer.eos_token_id]
            )

            if len(input_ids) > maximum_length:
                raise ValueError(
                    f"{record['id']} has "
                    f"{len(input_ids)} tokens; "
                    f"limit is {maximum_length}"
                )

            self.examples.append(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class CausalCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        maximum = max(
            len(example["input_ids"])
            for example in examples
        )
        maximum = math.ceil(maximum / 8) * 8

        input_ids = []
        attention_masks = []
        labels = []

        for example in examples:
            padding = (
                maximum
                - len(example["input_ids"])
            )

            input_ids.append(
                example["input_ids"]
                + [self.pad_token_id] * padding
            )
            attention_masks.append(
                [1] * len(example["input_ids"])
                + [0] * padding
            )
            labels.append(
                example["labels"]
                + [-100] * padding
            )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_masks,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
        }


def build_target_catalog(method, labels):
    if method == "dsi":
        pairs = [
            (
                label["leaf_id"],
                label["leaf_id"],
            )
            for label in labels
        ]
    else:
        pairs = [
            (
                label["canonical_path"],
                label["leaf_id"],
            )
            for label in labels
        ]

    return dict(pairs)


def build_token_trie(
    tokenizer,
    candidate_targets,
):
    trie = {}

    for target in candidate_targets:
        sequence = tokenizer(
            target,
            add_special_tokens=False,
        )["input_ids"]
        sequence.append(
            tokenizer.eos_token_id
        )

        node = trie

        for token_id in sequence:
            node = node.setdefault(
                token_id,
                {},
            )

    return trie


def evaluate_recall_at_1(
    model,
    tokenizer,
    records,
    template,
    toc,
    target_to_leaf,
    batch_size,
):
    candidate_targets = list(
        target_to_leaf
    )
    trie = build_token_trie(
        tokenizer,
        candidate_targets,
    )

    previous_padding_side = (
        tokenizer.padding_side
    )
    tokenizer.padding_side = "left"

    model.eval()
    model.config.use_cache = True

    correct = 0
    total = 0
    start = time.perf_counter()

    for offset in range(
        0,
        len(records),
        batch_size,
    ):
        batch_records = records[
            offset : offset + batch_size
        ]
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

        input_width = inputs[
            "input_ids"
        ].shape[1]

        def allowed_tokens(
            batch_id,
            input_ids,
        ):
            generated = input_ids[
                input_width:
            ].tolist()
            node = trie

            for token_id in generated:
                if token_id not in node:
                    return [
                        tokenizer.eos_token_id
                    ]
                node = node[token_id]

            allowed = list(node)

            if not allowed:
                return [
                    tokenizer.eos_token_id
                ]

            return allowed

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                num_beams=1,
                prefix_allowed_tokens_fn=(
                    allowed_tokens
                ),
                eos_token_id=(
                    tokenizer.eos_token_id
                ),
                pad_token_id=(
                    tokenizer.pad_token_id
                ),
            )

        generated = outputs[:, input_width:]
        predictions = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )

        for record, prediction in zip(
            batch_records,
            predictions,
        ):
            normalized = prediction.strip()

            if normalized not in target_to_leaf:
                raise ValueError(
                    "Constrained decoding produced "
                    f"unknown target: {normalized!r}"
                )

            predicted_leaf = target_to_leaf[
                normalized
            ]
            correct += (
                predicted_leaf
                == record["gold_leaf_id"]
            )
            total += 1

    model.config.use_cache = False
    model.train()
    tokenizer.padding_side = (
        previous_padding_side
    )

    return {
        "recall_at_1": correct / total,
        "correct": correct,
        "total": total,
        "seconds": (
            time.perf_counter() - start
        ),
    }


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    output_dir,
    state,
):
    last_adapter = (
        output_dir / "last-adapter"
    )
    model.save_pretrained(
        last_adapter,
        safe_serialization=True,
    )

    training_state_path = (
        output_dir / "training-state.pt"
    )
    temporary_state_path = (
        output_dir
        / "training-state.pt.tmp"
    )

    torch.save(
        {
            "optimizer": (
                optimizer.state_dict()
            ),
            "scheduler": (
                scheduler.state_dict()
            ),
        },
        temporary_state_path,
    )
    temporary_state_path.replace(
        training_state_path
    )

    state_path = (
        output_dir / "state.json"
    )
    temporary_json_path = (
        output_dir / "state.json.tmp"
    )
    temporary_json_path.write_text(
        json.dumps(
            state,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_json_path.replace(
        state_path
    )


def main():
    args = parse_args()
    set_seed(SEED)

    torch.backends.cuda.matmul.allow_tf32 = True

    defaults = DEFAULTS[args.method]
    batch_size = (
        args.batch_size
        or defaults["batch_size"]
    )
    gradient_accumulation = (
        args.gradient_accumulation
        or defaults[
            "gradient_accumulation"
        ]
    )
    eval_batch_size = (
        args.eval_batch_size
        or defaults["eval_batch_size"]
    )

    epochs = 1 if args.smoke else args.epochs

    output_name = (
        f"{args.method}-smoke"
        if args.smoke
        else args.method
    )
    output_dir = (
        Path("artifacts/models")
        / output_name
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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
    toc = (
        DATA_DIR / "toc.txt"
    ).read_text(
        encoding="utf-8"
    ).rstrip()

    train_records = read_jsonl(
        DATA_DIR
        / args.method
        / "train.jsonl"
    )
    dev_records = read_jsonl(
        DATA_DIR
        / args.method
        / "dev.jsonl"
    )

    if args.smoke:
        train_records = train_records[:16]
        dev_records = dev_records[:16]

    template = manifest[
        "prompt_templates"
    ][args.method]
    maximum_length = manifest[
        "maximum_sequence_lengths"
    ][args.method]

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_SNAPSHOT,
        local_files_only=True,
    )
    tokenizer.pad_token = (
        tokenizer.eos_token
    )
    tokenizer.padding_side = "right"

    train_dataset = RetrieverDataset(
        records=train_records,
        tokenizer=tokenizer,
        template=template,
        toc=toc,
        maximum_length=maximum_length,
    )
    collator = CausalCollator(
        tokenizer.pad_token_id
    )

    state_path = output_dir / "state.json"
    optimizer_state_path = (
        output_dir / "training-state.pt"
    )
    last_adapter = (
        output_dir / "last-adapter"
    )

    resume = (
        not args.fresh
        and state_path.exists()
        and optimizer_state_path.exists()
        and last_adapter.exists()
    )

    print(
        f"Loading {MODEL_ID} from "
        f"{MODEL_SNAPSHOT}"
    )
    load_start = time.perf_counter()

    base_model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_SNAPSHOT,
            local_files_only=True,
            use_safetensors=True,
            dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
        )
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        }
    )
    base_model.enable_input_require_grads()

    if resume:
        model = PeftModel.from_pretrained(
            base_model,
            last_adapter,
            is_trainable=True,
        )
    else:
        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=(
                LORA_TARGET_MODULES
            ),
        )
        model = get_peft_model(
            base_model,
            lora_config,
        )

    print(
        "Model load seconds: "
        f"{time.perf_counter() - load_start:.2f}"
    )
    model.print_trainable_parameters()
    model.train()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
    )

    batches_per_epoch = math.ceil(
        len(train_dataset) / batch_size
    )
    updates_per_epoch = math.ceil(
        batches_per_epoch
        / gradient_accumulation
    )
    total_updates = (
        updates_per_epoch * epochs
    )
    warmup_steps = max(
        1,
        round(total_updates * 0.1),
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_updates,
        )
    )

    if resume:
        state = json.loads(
            state_path.read_text(
                encoding="utf-8"
            )
        )
        checkpoint = torch.load(
            optimizer_state_path,
            map_location="cuda",
            weights_only=False,
        )
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )
        scheduler.load_state_dict(
            checkpoint["scheduler"]
        )
        start_epoch = state[
            "completed_epochs"
        ]
        best_recall = state[
            "best_dev_recall_at_1"
        ]
        epochs_without_improvement = (
            state[
                "epochs_without_improvement"
            ]
        )
        history = state["history"]

        print(
            f"Resuming after epoch "
            f"{start_epoch}"
        )
    else:
        start_epoch = 0
        best_recall = -1.0
        epochs_without_improvement = 0
        history = []

    target_to_leaf = build_target_catalog(
        args.method,
        labels,
    )

    run_config = {
        "method": args.method,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
        "epochs": epochs,
        "learning_rate": (
            args.learning_rate
        ),
        "batch_size": batch_size,
        "gradient_accumulation": (
            gradient_accumulation
        ),
        "effective_batch_size": (
            batch_size
            * gradient_accumulation
        ),
        "eval_batch_size": (
            eval_batch_size
        ),
        "maximum_length": maximum_length,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lora_target_modules": (
            LORA_TARGET_MODULES
        ),
        "train_examples": len(
            train_records
        ),
        "dev_examples": len(dev_records),
        "early_stopping_metric": (
            "dev_recall_at_1"
        ),
        "patience": args.patience,
    }

    (output_dir / "run-config.json").write_text(
        json.dumps(
            run_config,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"Method: {args.method}"
    )
    print(
        f"Train/dev: "
        f"{len(train_records)}/"
        f"{len(dev_records)}"
    )
    print(
        "Effective batch size: "
        f"{batch_size * gradient_accumulation}"
    )
    print(
        f"Updates per epoch: "
        f"{updates_per_epoch}"
    )

    for epoch in range(
        start_epoch,
        epochs,
    ):
        generator = torch.Generator()
        generator.manual_seed(SEED + epoch)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collator,
            num_workers=0,
        )

        model.train()
        model.config.use_cache = False
        optimizer.zero_grad(
            set_to_none=True
        )
        torch.cuda.reset_peak_memory_stats()

        loss_sum = 0.0
        epoch_start = time.perf_counter()

        for batch_index, batch in enumerate(
            train_loader,
            start=1,
        ):
            batch = {
                key: value.to(
                    "cuda",
                    non_blocking=True,
                )
                for key, value in batch.items()
            }

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                output = model(**batch)
                loss = (
                    output.loss
                    / gradient_accumulation
                )

            loss.backward()
            loss_sum += output.loss.item()

            should_update = (
                batch_index
                % gradient_accumulation
                == 0
                or batch_index
                == len(train_loader)
            )

            if should_update:
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(
                    set_to_none=True
                )

            if (
                batch_index % 25 == 0
                or batch_index
                == len(train_loader)
            ):
                print(
                    f"Epoch {epoch + 1}/"
                    f"{epochs} | "
                    f"batch {batch_index}/"
                    f"{len(train_loader)} | "
                    f"loss "
                    f"{output.loss.item():.4f}",
                    flush=True,
                )

        training_seconds = (
            time.perf_counter()
            - epoch_start
        )
        average_loss = (
            loss_sum / len(train_loader)
        )
        peak_gpu_gib = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        dev_metrics = evaluate_recall_at_1(
            model=model,
            tokenizer=tokenizer,
            records=dev_records,
            template=template,
            toc=toc,
            target_to_leaf=target_to_leaf,
            batch_size=eval_batch_size,
        )
        dev_recall = dev_metrics[
            "recall_at_1"
        ]

        improved = (
            dev_recall > best_recall
        )

        if improved:
            best_recall = dev_recall
            epochs_without_improvement = 0
            model.save_pretrained(
                output_dir / "best-adapter",
                safe_serialization=True,
            )
        else:
            epochs_without_improvement += 1

        epoch_record = {
            "epoch": epoch + 1,
            "average_train_loss": (
                average_loss
            ),
            "dev_recall_at_1": dev_recall,
            "dev_correct": dev_metrics[
                "correct"
            ],
            "dev_total": dev_metrics[
                "total"
            ],
            "training_seconds": (
                training_seconds
            ),
            "dev_evaluation_seconds": (
                dev_metrics["seconds"]
            ),
            "peak_gpu_memory_gib": (
                peak_gpu_gib
            ),
            "learning_rate": (
                scheduler.get_last_lr()[0]
            ),
            "improved": improved,
        }
        history.append(epoch_record)

        state = {
            "completed_epochs": epoch + 1,
            "best_dev_recall_at_1": (
                best_recall
            ),
            "epochs_without_improvement": (
                epochs_without_improvement
            ),
            "history": history,
            "complete": False,
        }

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            output_dir=output_dir,
            state=state,
        )

        print(
            f"Epoch {epoch + 1} complete | "
            f"loss={average_loss:.4f} | "
            f"dev Recall@1="
            f"{dev_recall:.4f} | "
            f"best={best_recall:.4f} | "
            f"peak={peak_gpu_gib:.2f} GiB",
            flush=True,
        )

        if (
            epochs_without_improvement
            >= args.patience
        ):
            print(
                "Early stopping: development "
                "Recall@1 did not improve"
            )
            break

    state["complete"] = True
    state_path.write_text(
        json.dumps(
            state,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Training complete")
    print(
        "Best development Recall@1: "
        f"{best_recall:.4f}"
    )
    print(
        "Best adapter: "
        f"{output_dir / 'best-adapter'}"
    )


if __name__ == "__main__":
    main()
