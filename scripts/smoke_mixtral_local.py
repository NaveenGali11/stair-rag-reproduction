from pathlib import Path
import json
import os
from huggingface_hub import snapshot_download


# SageMaker has the CUDA runtime but not the development headers that
# FlashInfer's optional sampling JIT requires. Use vLLM's native sampler.
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

from vllm import LLM, SamplingParams


MODEL_ID = "casperhansen/mixtral-instruct-awq"
MODEL_REVISION = "0a898130957afe22021bbaf807f50f6bbce88201"

MODEL_PATH = Path(
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
    )
)

PARAGRAPHS_PATH = Path("data/derived/whole-child-paragraphs.json")
PARAGRAPH_ID = "whole-child-023-p001"


if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model snapshot not found: {MODEL_PATH}")

paragraphs = json.loads(PARAGRAPHS_PATH.read_text(encoding="utf-8"))
record = next(item for item in paragraphs if item["id"] == PARAGRAPH_ID)

instruction = f"""
You create evaluation questions from a textbook paragraph.

Write exactly three standalone information-seeking questions.

Requirements:
- Every question must be answerable using only the paragraph.
- Cover distinct important claims where possible.
- Do not mention "the paragraph", "the passage", the author, or a chapter.
- Do not include answers or explanations.
- Return only a JSON array of three strings.

Paragraph:
{record["text"]}
""".strip()

# Mixtral Instruct was trained with this instruction wrapper.
prompt = f"[INST] {instruction} [/INST]"

print(f"Model revision: {MODEL_REVISION}")
print(f"Paragraph: {record['id']}")
print(f"Gold leaf: {record['leaf_id']}")
print(f"Characters: {record['character_count']}")
print("\nLoading model...")

model = LLM(
    model=str(MODEL_PATH),
    quantization="awq",
    dtype="float16",
    tensor_parallel_size=1,
    max_model_len=4096,
    max_num_seqs=1,
    gpu_memory_utilization=0.80,
    enforce_eager=True,
    trust_remote_code=False,
    seed=42,
)

sampling = SamplingParams(
    temperature=0.0,
    max_tokens=256,
    seed=42,
)

outputs = model.generate(
    [prompt],
    sampling,
    use_tqdm=True,
)

raw_output = outputs[0].outputs[0].text.strip()

print("\nRaw model output:\n")
print(raw_output)

questions = json.loads(raw_output)

if not isinstance(questions, list):
    raise ValueError("Model output is not a JSON array")

if len(questions) != 3:
    raise ValueError(f"Expected 3 questions, received {len(questions)}")

if not all(isinstance(question, str) for question in questions):
    raise ValueError("Every question must be a string")

print("\nValidated questions:\n")

for index, question in enumerate(questions, start=1):
    print(f"{index}. {question}")