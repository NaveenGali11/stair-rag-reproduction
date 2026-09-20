from pathlib import Path
import json
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

MODEL_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/"
    "models--mistralai--Mistral-7B-Instruct-v0.2/"
    "snapshots/"
    "63a8b081895390a26e140280378bc85ec8bce07a"
)
DATA_DIR = Path("data/retriever/whole-child")

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def read_first(path):
    line = next(
        line
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    )
    return json.loads(line)


manifest = json.loads(
    (DATA_DIR / "manifest.json").read_text(
        encoding="utf-8"
    )
)
toc = (DATA_DIR / "toc.txt").read_text(
    encoding="utf-8"
).rstrip()

dsi_record = read_first(
    DATA_DIR / "dsi/train.jsonl"
)
stair_record = read_first(
    DATA_DIR / "stair/train.jsonl"
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_SNAPSHOT,
    local_files_only=True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"Loading model from {MODEL_SNAPSHOT}")
load_start = time.perf_counter()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_SNAPSHOT,
    local_files_only=True,
    use_safetensors=True,
    dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="sdpa",
)

print(
    "Base-model load seconds: "
    f"{time.perf_counter() - load_start:.2f}"
)

model.config.use_cache = False
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={
        "use_reentrant": False,
    }
)
model.enable_input_require_grads()

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=LORA_TARGET_MODULES,
)

model = get_peft_model(
    model,
    lora_config,
)
model.print_trainable_parameters()
model.train()

optimizer = torch.optim.AdamW(
    (
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ),
    lr=2e-4,
)


def build_example(method, record):
    template = manifest[
        "prompt_templates"
    ][method]

    prompt = template.format(
        query=record["query"],
        toc=toc,
    )

    prompt_ids = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        tokenize=True,
        add_generation_prompt=True,
    )

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

    maximum = manifest[
        "maximum_sequence_lengths"
    ][method]

    if len(input_ids) > maximum:
        raise ValueError(
            f"{method} example has "
            f"{len(input_ids)} tokens; "
            f"limit is {maximum}"
        )

    return {
        "input_ids": torch.tensor(
            [input_ids],
            dtype=torch.long,
            device="cuda",
        ),
        "attention_mask": torch.ones(
            (1, len(input_ids)),
            dtype=torch.long,
            device="cuda",
        ),
        "labels": torch.tensor(
            [labels],
            dtype=torch.long,
            device="cuda",
        ),
    }


for method, record in (
    ("dsi", dsi_record),
    ("stair", stair_record),
):
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    batch = build_example(method, record)

    start = time.perf_counter()
    output = model(**batch)
    output.loss.backward()
    optimizer.step()
    elapsed = time.perf_counter() - start

    print(
        f"{method.upper()} smoke passed"
    )
    print(
        f"  tokens={batch['input_ids'].shape[1]}"
    )
    print(f"  loss={output.loss.item():.4f}")
    print(f"  step_seconds={elapsed:.2f}")
    print(
        "  peak_gpu_gib="
        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f}"
    )

print("Retriever LoRA smoke test passed")
