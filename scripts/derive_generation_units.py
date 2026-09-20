from collections import Counter, defaultdict
from pathlib import Path
import json
import re
import statistics


INPUT_PATH = Path(
    "data/derived/whole-child-paragraphs.json"
)
UNITS_PATH = Path(
    "data/derived/whole-child-generation-units.json"
)
REMOVALS_PATH = Path(
    "data/derived/whole-child-generation-removals.json"
)

MAX_LIST_CHARS = 3000
SHORT_FRAGMENT_CHARS = 40
SHORT_LABEL_CHARS = 60

INTERACTIVE_ELEMENT_PLACEHOLDER_PATTERN = re.compile(
    r"^(?:an|one or more)\s+interactive"
    r"(?:\s+H5P)?\s+elements?\s+has\s+been"
    r"\s+excluded\s+from\s+this\s+version"
    r"\s+of\s+the\s+text\.",
    re.IGNORECASE,
)

URL_ONLY_PATTERN = re.compile(
    r"^\s*(?:[•◦*-]\s*)?"
    r"(?:(?:online\s+here|url)\s*:\s*)?"
    r"\(?https?://\S+\)?\s*$",
    re.IGNORECASE,
)

ORPHAN_URL_TAIL_PATTERN = re.compile(
    r"""^[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+$"""
)

AGE_BAND_PATTERN = re.compile(
    r"^\d+\s*[–—-]\s*\d+"
    r"(?:\s+months?)?$",
    re.IGNORECASE,
)

PARENT_BULLET_PATTERN = re.compile(
    r"^[•●*-]\s+"
)

NESTED_BULLET_PATTERN = re.compile(
    r"^[◦▪]\s*(?:o\s+)?"
)

FORWARD_LABELS = {
    "did you know…",
    "months milestones",
}

CITATION_ONLY_PATTERN = re.compile(
    r"^\(.*\)\.?$"
)

def ends_sentence(text: str) -> bool:
    return bool(
        re.search(
            r"""[.!?…][”"'’)\]]*$""",
            text.strip(),
        )
    )


def starts_parent_bullet(text: str) -> bool:
    return bool(
        PARENT_BULLET_PATTERN.match(text.strip())
    )


def starts_nested_bullet(text: str) -> bool:
    return bool(
        NESTED_BULLET_PATTERN.match(text.strip())
    )


def starts_any_bullet(text: str) -> bool:
    return (
        starts_parent_bullet(text)
        or starts_nested_bullet(text)
    )

def contains_any_bullet(text: str) -> bool:
    return any(
        symbol in text
        for symbol in ("•", "●", "◦", "▪")
    )


def starts_lowercase_prose(text: str) -> bool:
    if starts_any_bullet(text):
        return False

    for character in text.strip():
        if character.isalpha():
            return character.islower()

    return False

def is_age_band(text: str) -> bool:
    return bool(
        AGE_BAND_PATTERN.fullmatch(text.strip())
    )


def is_orphan_url_tail(text: str) -> bool:
    stripped = text.strip()

    return (
        len(stripped) <= SHORT_FRAGMENT_CHARS
        and "/" in stripped
        and bool(
            ORPHAN_URL_TAIL_PATTERN.fullmatch(
                stripped
            )
        )
    )


def make_unit(paragraph):
    return {
        "leaf_id": paragraph["leaf_id"],
        "leaf_title": paragraph["leaf_title"],
        "path": paragraph["path"],
        "page_start": paragraph["page_start"],
        "page_end": paragraph["page_end"],
        "source_paragraph_ids": [paragraph["id"]],
        "merge_reasons": [],
        "text": paragraph["text"],
        "character_count": len(paragraph["text"]),
    }


def merge_units(left, right, reason):
    if left["leaf_id"] != right["leaf_id"]:
        raise ValueError(
            "Attempted cross-leaf merge: "
            f"{left['leaf_id']} and {right['leaf_id']}"
        )

    separator = (
        "\n"
        if starts_any_bullet(right["text"])
        else " "
    )

    text = (
        left["text"].rstrip()
        + separator
        + right["text"].lstrip()
    )

    return {
        "leaf_id": left["leaf_id"],
        "leaf_title": left["leaf_title"],
        "path": left["path"],
        "page_start": min(
            left["page_start"],
            right["page_start"],
        ),
        "page_end": max(
            left["page_end"],
            right["page_end"],
        ),
        "source_paragraph_ids": [
            *left["source_paragraph_ids"],
            *right["source_paragraph_ids"],
        ],
        "merge_reasons": [
            *left["merge_reasons"],
            *right["merge_reasons"],
            reason,
        ],
        "text": text,
        "character_count": len(text),
    }


def remove_noise(group):
    survivors = []
    removals = []
    in_attribution_tail = False
    previous_was_url_only = False

    for paragraph in group:
        text = paragraph["text"].strip()

        if text.casefold() == "media attributions":
            in_attribution_tail = True

        if in_attribution_tail:
            reason = "media_attribution_tail"
        elif INTERACTIVE_ELEMENT_PLACEHOLDER_PATTERN.match(text):
            reason = "interactive_element_placeholder"
        elif URL_ONLY_PATTERN.fullmatch(text):
            reason = "url_only"
        elif (
            previous_was_url_only
            and is_orphan_url_tail(text)
        ):
            reason = "orphan_url_tail"
        else:
            reason = None

        if reason is not None:
            removals.append(
                {
                    "source_paragraph_id": paragraph["id"],
                    "leaf_id": paragraph["leaf_id"],
                    "page_start": paragraph["page_start"],
                    "page_end": paragraph["page_end"],
                    "reason": reason,
                    "text": paragraph["text"],
                }
            )

            previous_was_url_only = (
                reason in {
                    "url_only",
                    "orphan_url_tail",
                }
            )
            continue

        previous_was_url_only = False
        survivors.append(make_unit(paragraph))

    return survivors, removals


def merge_backward_fragments(units):
    result = []

    for unit in units:
        text = unit["text"].strip()

        should_merge = (
            bool(result)
            and len(text) <= SHORT_FRAGMENT_CHARS
            and not starts_any_bullet(text)
            and not is_age_band(text)
            and not ends_sentence(
                result[-1]["text"]
            )
        )

        if should_merge:
            result[-1] = merge_units(
                result[-1],
                unit,
                "short_continuation_backward",
            )
        else:
            result.append(unit)

    return result


def merge_age_bands_forward(units):
    result = []
    index = 0

    while index < len(units):
        unit = units[index]

        if (
            is_age_band(unit["text"])
            and index + 1 < len(units)
        ):
            result.append(
                merge_units(
                    unit,
                    units[index + 1],
                    "age_band_forward",
                )
            )
            index += 2
        else:
            result.append(unit)
            index += 1

    return result


def is_forward_label(unit) -> bool:
    text = unit["text"].strip()
    folded = text.casefold()

    if folded in FORWARD_LABELS:
        return True

    return (
        len(text) <= SHORT_LABEL_CHARS
        and not starts_any_bullet(text)
        and not ends_sentence(text)
    )


def merge_labels_forward(units):
    result = []
    index = 0

    while index < len(units):
        unit = units[index]

        if (
            is_forward_label(unit)
            and index + 1 < len(units)
        ):
            result.append(
                merge_units(
                    unit,
                    units[index + 1],
                    "short_label_forward",
                )
            )
            index += 2
        else:
            result.append(unit)
            index += 1

    return result


def group_bullet_lists(units):
    result = []

    for unit in units:
        text = unit["text"].strip()

        if not result:
            result.append(unit)
            continue

        combined_length = (
            len(result[-1]["text"])
            + 1
            + len(unit["text"])
        )

        nested_bullet = starts_nested_bullet(text)

        adjacent_parent_bullets = (
            starts_parent_bullet(text)
            and starts_parent_bullet(
                result[-1]["text"]
            )
        )

        should_merge = (
            combined_length <= MAX_LIST_CHARS
            and (
                nested_bullet
                or adjacent_parent_bullets
            )
        )

        if should_merge:
            reason = (
                "nested_bullet_backward"
                if nested_bullet
                else "adjacent_bullets"
            )

            result[-1] = merge_units(
                result[-1],
                unit,
                reason,
            )
        else:
            result.append(unit)

    return result

def merge_artifact_continuations(units):
    result = []

    for unit in units:
        if not result:
            result.append(unit)
            continue

        current_text = unit["text"].strip()
        combined_length = (
            len(result[-1]["text"])
            + 1
            + len(unit["text"])
        )

        should_merge = (
            combined_length <= MAX_LIST_CHARS
            and (
                CITATION_ONLY_PATTERN.fullmatch(
                    current_text
                )
                or starts_lowercase_prose(
                    current_text
                )
            )
        )

        if should_merge:
            result[-1] = merge_units(
                result[-1],
                unit,
                "artifact_continuation_backward",
            )
        else:
            result.append(unit)

    return result


def complete_label_sentences(units):
    result = []
    index = 0

    while index < len(units):
        unit = units[index]

        while (
            "short_label_forward"
            in unit["merge_reasons"]
            and not contains_any_bullet(
                unit["text"]
            )
            and not ends_sentence(
                unit["text"]
            )
            and index + 1 < len(units)
        ):
            following = units[index + 1]
            combined_length = (
                len(unit["text"])
                + 1
                + len(following["text"])
            )

            if combined_length > MAX_LIST_CHARS:
                break

            unit = merge_units(
                unit,
                following,
                "label_sentence_completion",
            )
            index += 1

        result.append(unit)
        index += 1

    return result

paragraphs = json.loads(
    INPUT_PATH.read_text(encoding="utf-8")
)

paragraphs_by_leaf = defaultdict(list)

for paragraph in paragraphs:
    paragraphs_by_leaf[
        paragraph["leaf_id"]
    ].append(paragraph)

generation_units = []
removals = []

for leaf_id, group in paragraphs_by_leaf.items():
    units, leaf_removals = remove_noise(group)
    removals.extend(leaf_removals)

    units = merge_backward_fragments(units)
    units = merge_age_bands_forward(units)
    units = merge_labels_forward(units)
    units = group_bullet_lists(units)
    units = merge_artifact_continuations(units)
    units = complete_label_sentences(units)

    for number, unit in enumerate(units, start=1):
        generation_units.append(
            {
                "id": (
                    f"{leaf_id}-g{number:03d}"
                ),
                **unit,
            }
        )

raw_ids = {
    paragraph["id"]
    for paragraph in paragraphs
}

retained_ids = {
    source_id
    for unit in generation_units
    for source_id in unit[
        "source_paragraph_ids"
    ]
}

removed_ids = {
    removal["source_paragraph_id"]
    for removal in removals
}

if retained_ids & removed_ids:
    raise ValueError(
        "A source paragraph was both retained "
        "and removed"
    )

if retained_ids | removed_ids != raw_ids:
    raise ValueError(
        "Source-paragraph accounting mismatch"
    )

represented_leaves = {
    unit["leaf_id"]
    for unit in generation_units
}

if len(represented_leaves) != 129:
    raise ValueError(
        "Expected 129 represented leaves, found "
        f"{len(represented_leaves)}"
    )

UNITS_PATH.write_text(
    json.dumps(
        generation_units,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

REMOVALS_PATH.write_text(
    json.dumps(
        removals,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

removal_counts = Counter(
    removal["reason"]
    for removal in removals
)

merge_counts = Counter(
    reason
    for unit in generation_units
    for reason in unit["merge_reasons"]
)

lengths = [
    unit["character_count"]
    for unit in generation_units
]

print(f"Raw paragraphs: {len(paragraphs)}")
print(f"Removed paragraphs: {len(removals)}")
print(f"Retained source paragraphs: {len(retained_ids)}")
print(f"Generation units: {len(generation_units)}")
print(f"Represented leaves: {len(represented_leaves)}")

print("\nRemoval counts:")
for reason, count in sorted(removal_counts.items()):
    print(f"- {reason}: {count}")

print("\nMerge counts:")
for reason, count in sorted(merge_counts.items()):
    print(f"- {reason}: {count}")

print(
    "\nMedian characters per generation unit: "
    f"{statistics.median(lengths):.0f}"
)
print(
    "Shortest generation unit: "
    f"{min(lengths)}"
)
print(
    "Longest generation unit: "
    f"{max(lengths)}"
)

print(f"\nWrote: {UNITS_PATH}")
print(f"Wrote: {REMOVALS_PATH}")
