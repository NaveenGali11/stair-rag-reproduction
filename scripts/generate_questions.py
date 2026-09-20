from pathlib import Path
import argparse
import json
import os


os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

INPUT_PATH = Path(
    "data/derived/whole-child-generation-units.json"
)
DEFAULT_OUTPUT_PATH = Path(
    "artifacts/generated/"
    "whole-child-mixtral-questions.jsonl"
)

MODEL_ID = "casperhansen/mixtral-instruct-awq"
MODEL_REVISION = (
    "0a898130957afe22021bbaf807f50f6bbce88201"
)
PROMPT_VERSION = "whole-child-questions-v1"
SEED = 42


def requested_question_count(character_count):
    if character_count <= 100:
        return 2
    if character_count <= 250:
        return 4
    if character_count <= 500:
        return 6
    if character_count <= 1000:
        return 8
    return 10


def build_prompt(text, count):
    return f"""[INST]
Generate exactly {count} standalone information-seeking questions from the passage below.

Requirements:
- Every question must be answerable using only the passage.
- Cover distinct factual or conceptual claims.
- Do not mention "the passage", "the text", an author, a chapter, or a section.
- Do not include answers or explanations.
- Avoid duplicate or near-duplicate questions.
- Return only a valid JSON array of exactly {count} strings.

Passage:
{text}
[/INST]"""


def parse_questions(raw_text, expected_count):
    start = raw_text.find("[")
    end = raw_text.rfind("]")

    if start == -1 or end == -1 or end < start:
        raise ValueError("No complete JSON array found")

    candidate = raw_text[start : end + 1]
    candidate = candidate.replace("•", "")

    try:
        questions = json.loads(candidate)
    except json.JSONDecodeError:
        questions = []
        decoder = json.JSONDecoder()
        cursor = 0

        while cursor < len(candidate):
            quote = candidate.find('"', cursor)

            if quote == -1:
                break

            try:
                value, consumed = decoder.raw_decode(
                    candidate[quote:]
                )
            except json.JSONDecodeError:
                cursor = quote + 1
                continue

            if isinstance(value, str):
                questions.append(value)

            cursor = quote + consumed

    while (
        isinstance(questions, list)
        and len(questions) == 1
        and isinstance(questions[0], list)
    ):
        questions = questions[0]

    if not isinstance(questions, list):
        raise ValueError("Output is not a list")

    cleaned = []
    seen = set()

    for item in questions:
        if isinstance(item, str):
            question = item
        elif isinstance(item, dict):
            question = item.get("question")

            if not isinstance(question, str):
                continue
        else:
            continue

        question = " ".join(question.split())

        if not question or not question.endswith("?"):
            continue

        key = question.casefold()

        if key in seen:
            continue

        seen.add(key)
        cleaned.append(question)

        if len(cleaned) == expected_count:
            break

    if len(cleaned) < 2:
        raise ValueError(
            "Fewer than two valid questions remained "
            f"after normalization: {len(cleaned)}"
        )

    return cleaned


def read_completed(output_path):
    if not output_path.exists():
        return {}

    completed = {}

    with output_path.open(
        encoding="utf-8"
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            if not line.strip():
                continue

            record = json.loads(line)

            if (
                record["model_id"] != MODEL_ID
                or record["model_revision"]
                != MODEL_REVISION
                or record["prompt_version"]
                != PROMPT_VERSION
            ):
                raise ValueError(
                    "Existing output metadata mismatch "
                    f"on line {line_number}"
                )

            unit_id = record["unit_id"]

            if unit_id in completed:
                raise ValueError(
                    f"Duplicate completed unit: {unit_id}"
                )

            completed[unit_id] = record

    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--limit",
        type=int,
    )
    parser.add_argument(
        "--plan",
        action="store_true",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    args = parser.parse_args()

    units = json.loads(
        INPUT_PATH.read_text(encoding="utf-8")
    )

    completed = read_completed(args.output)

    pending = [
        unit
        for unit in units
        if unit["id"] not in completed
    ]

    if args.limit is not None:
        pending = pending[: args.limit]

    planned_questions = sum(
        requested_question_count(
            unit["character_count"]
        )
        for unit in units
    )

    pending_questions = sum(
        requested_question_count(
            unit["character_count"]
        )
        for unit in pending
    )

    print(f"Generation units: {len(units)}")
    print(f"Already completed: {len(completed)}")
    print(f"Pending this run: {len(pending)}")
    print(f"Planned total questions: {planned_questions}")
    print(
        "Planned questions this run: "
        f"{pending_questions}"
    )

    if args.plan or not pending:
        return

    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams

    model_path = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
    )

    print(f"Loading model from: {model_path}")

    model = LLM(
        model=model_path,
        quantization="awq",
        dtype="float16",
        tensor_parallel_size=1,
        max_model_len=4096,
        max_num_seqs=args.batch_size,
        gpu_memory_utilization=0.80,
        enforce_eager=True,
        trust_remote_code=False,
        seed=SEED,
        safetensors_load_strategy="prefetch"
    )

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        seed=SEED,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for batch_start in range(
        0,
        len(pending),
        args.batch_size,
    ):
        batch = pending[
            batch_start :
            batch_start + args.batch_size
        ]

        prompts = [
            build_prompt(
                unit["text"],
                requested_question_count(
                    unit["character_count"]
                ),
            )
            for unit in batch
        ]

        outputs = model.generate(
            prompts,
            sampling,
            use_tqdm=True,
        )

        records = []

        for unit, output in zip(batch, outputs):
            expected_count = (
                requested_question_count(
                    unit["character_count"]
                )
            )
            raw_text = output.outputs[0].text
            try:
                questions = parse_questions(
                    raw_text,
                    expected_count,
                )
            except ValueError as error:
                failure_path = args.output.with_suffix(
                    ".failures.jsonl"
                )
                failure_record = {
                    "unit_id": unit["id"],
                    "leaf_id": unit["leaf_id"],
                    "requested_question_count": expected_count,
                    "error": str(error),
                    "raw_output": raw_text,
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "prompt_version": PROMPT_VERSION,
                    "seed": SEED,
                }

                with failure_path.open(
                    "a",
                    encoding="utf-8",
                ) as failure_handle:
                    print(
                        json.dumps(
                            failure_record,
                            ensure_ascii=False,
                        ),
                        file=failure_handle,
                        flush=True,
                    )
                    os.fsync(
                        failure_handle.fileno()
                    )

                print(
                    f"Quarantined {unit['id']}: "
                    f"{error}",
                    flush=True,
                )
                continue

            if len(questions) != expected_count:
              print(
                  f"Accepted {len(questions)}/"
                  f"{expected_count} questions for "
                  f"{unit['id']}",
                  flush=True,
              )

            records.append(
                {
                    "unit_id": unit["id"],
                    "leaf_id": unit["leaf_id"],
                    "source_paragraph_ids": (
                        unit[
                            "source_paragraph_ids"
                        ]
                    ),
                    "requested_question_count": (
                        expected_count
                    ),
                    "questions": questions,
                    "model_id": MODEL_ID,
                    "model_revision": (
                        MODEL_REVISION
                    ),
                    "prompt_version": (
                        PROMPT_VERSION
                    ),
                    "seed": SEED,
                }
            )

        with args.output.open(
            "a",
            encoding="utf-8",
        ) as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            handle.flush()
            os.fsync(handle.fileno())

        finished = (
            len(completed)
            + batch_start
            + len(batch)
        )

        print(
            f"Checkpointed {finished}/"
            f"{len(units)} units"
        )


if __name__ == "__main__":
    main()