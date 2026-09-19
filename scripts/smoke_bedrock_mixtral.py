from pathlib import Path
import json
import re

import boto3


MODEL_ID = "mistral.mixtral-8x7b-instruct-v0:1"
REGION = "us-east-1"
PARAGRAPHS_PATH = Path(
    "data/derived/whole-child-paragraphs.json"
)
PARAGRAPH_ID = "whole-child-023-p001"

paragraphs = json.loads(
    PARAGRAPHS_PATH.read_text(encoding="utf-8")
)
paragraph = next(
    item for item in paragraphs
    if item["id"] == PARAGRAPH_ID
)

prompt = f"""
Create natural information-seeking questions for retrieval evaluation.

Using only the textbook paragraph below, generate up to three questions.
Each question must:

- Be answerable using only the paragraph.
- Cover an important idea from the paragraph.
- Be understandable without seeing the paragraph.
- Ask for information rather than mention the book, paragraph, or section.
- Be distinct from the other questions.
- End with a question mark.

Do not provide answers or explanations.

Return exactly one JSON object in this form:
{{"questions": ["question one?", "question two?"]}}

Textbook paragraph:
{paragraph["text"]}
""".strip()

client = boto3.client(
    "bedrock-runtime",
    region_name=REGION,
)

response = client.converse(
    modelId=MODEL_ID,
    messages=[
        {
            "role": "user",
            "content": [{"text": prompt}],
        }
    ],
    inferenceConfig={
        "maxTokens": 300,
        "temperature": 0.0,
        "topP": 1.0,
    },
)

raw_output = "".join(
    block.get("text", "")
    for block in response["output"]["message"]["content"]
).strip()

json_text = re.sub(
    r"^```(?:json)?\s*|\s*```$",
    "",
    raw_output,
    flags=re.IGNORECASE,
)

payload = json.loads(json_text)
questions = payload["questions"]

if not 1 <= len(questions) <= 3:
    raise ValueError(
        f"Expected 1–3 questions, received {len(questions)}"
    )

if not all(
    isinstance(question, str)
    and question.strip().endswith("?")
    for question in questions
):
    raise ValueError(f"Invalid questions: {questions}")

print(f"Model: {MODEL_ID}")
print(f"Paragraph: {PARAGRAPH_ID}")
print(f"Usage: {response.get('usage')}")
print("\nRaw output:")
print(raw_output)

print("\nValidated questions:")
for number, question in enumerate(questions, start=1):
    print(f"{number}. {question}")