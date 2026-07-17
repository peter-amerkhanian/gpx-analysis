import pandas as pd

from .constants import ROAD_NAME_SUFFIXES


def _normalize_match_text(value: object) -> str | None:
    """Return a lowercased, trimmed string for fuzzy name comparisons."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    return text or None


def _road_name_key(value: object) -> str | None:
    """Return a normalized key that treats common road suffix variants as equal."""
    text = _normalize_match_text(value)
    if text is None:
        return None
    tokens = [
        token
        for token in "".join(char if char.isalnum() else " " for char in text).split()
        if token and token not in ROAD_NAME_SUFFIXES
    ]
    if not tokens:
        return text
    return " ".join(tokens)


def _levenshtein_distance(left: str | None, right: str | None) -> int | None:
    """Return the Levenshtein distance between two normalized strings."""
    left = _normalize_match_text(left)
    right = _normalize_match_text(right)
    if left is None or right is None:
        return None
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]
