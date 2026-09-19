from collections import Counter
from pathlib import Path
import json

leaves_path = Path("data/derived/whole-child-retrieval-leaves.json")
sections_path = Path("data/derived/whole-child-sections.json")

leaves = json.loads(leaves_path.read_text(encoding="utf-8"))
sections = json.loads(sections_path.read_text(encoding="utf-8"))

expected_ids = {leaf["id"] for leaf in leaves}
actual_ids = [section["id"] for section in sections]
actual_id_set = set(actual_ids)

missing_ids = expected_ids - actual_id_set
extra_ids = actual_id_set - expected_ids
duplicate_ids = [
    section_id
    for section_id, count in Counter(actual_ids).items()
    if count > 1
]
empty_sections = [
    section for section in sections if not section["text"].strip()
]

if missing_ids or extra_ids or duplicate_ids or empty_sections:
    raise ValueError(
        {
            "missing_ids": sorted(missing_ids),
            "extra_ids": sorted(extra_ids),
            "duplicate_ids": duplicate_ids,
            "empty_sections": [section["id"] for section in empty_sections],
        }
    )

print("Section validation passed")
print(f"Expected leaves: {len(expected_ids)}")
print(f"Extracted sections: {len(sections)}")

print("\nFive shortest sections:")
for section in sorted(sections, key=lambda item: item["character_count"])[:5]:
    print(
        f"- {section['character_count']:>4} chars | "
        f"{' > '.join(section['path'])}"
    )

print("\nFive longest sections:")
for section in sorted(
    sections,
    key=lambda item: item["character_count"],
    reverse=True,
)[:5]:
    print(
        f"- {section['character_count']:>4} chars | "
        f"{' > '.join(section['path'])}"
    )