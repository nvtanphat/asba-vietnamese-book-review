from __future__ import annotations

from collections import Counter
import unicodedata
from typing import Any, Mapping, Sequence

import regex as re

from .helpers import percentage, to_dataframe

TOKEN_RE = re.compile(r"(?V1)\p{L}[\p{L}\p{M}\p{N}_'-]*")
REPEAT_RE = re.compile(r"(.)\1{1,}")
ASCII_WORD_RE = re.compile(r"^[A-Za-z]+$")
MIXED_ALNUM_RE = re.compile(r"(?=.*\p{L})(?=.*\p{N})")
VOWEL_RE = re.compile(r"[aeiouyAEIOUY]")


def _strip_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _token_features(token: str) -> dict[str, bool]:
    ascii_only = token.isascii() and ASCII_WORD_RE.fullmatch(token) is not None
    stripped = _strip_diacritics(token)
    accentless = stripped.casefold() == token.casefold()
    short = len(token) <= 3
    repeated = REPEAT_RE.search(token) is not None
    mixed_alnum = MIXED_ALNUM_RE.search(token) is not None
    lowercase_ascii = ascii_only and token.islower()
    uppercase = token.isupper() and len(token) > 1
    no_vowel = ascii_only and VOWEL_RE.search(token) is None
    return {
        "ascii_only": ascii_only,
        "accentless": accentless,
        "short": short,
        "repeated": repeated,
        "mixed_alnum": mixed_alnum,
        "lowercase_ascii": lowercase_ascii,
        "uppercase": uppercase,
        "no_vowel": no_vowel,
    }


def _counter_to_list(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"token": token, "count": count}
        for token, count in counter.most_common()
    ]


def scan(records: Sequence[Mapping[str, Any]], text_column: str | None = None) -> dict[str, Any]:
    if not records or not text_column:
        return {
            "text_column": text_column,
            "token_stats": {},
            "rows_with_suspicious_vocab": 0,
            "rows_with_suspicious_vocab_ratio": 0.0,
            "teencode_like_tokens": [],
            "elongated_tokens": [],
            "accentless_ascii_tokens": [],
            "mixed_alnum_tokens": [],
            "possible_misspellings": [],
        }

    df = to_dataframe(records)
    series = df[text_column].fillna("").astype(str)
    token_counter = Counter()
    teencode_like_tokens: Counter[str] = Counter()
    elongated_tokens: Counter[str] = Counter()
    accentless_ascii_tokens: Counter[str] = Counter()
    mixed_alnum_tokens: Counter[str] = Counter()
    possible_misspellings: Counter[str] = Counter()
    rows_with_suspicious_vocab = 0

    for text in series.tolist():
        tokens = TOKEN_RE.findall(text)
        if not tokens:
            continue

        token_counter.update(tokens)
        row_has_suspicious = False

        for token in tokens:
            features = _token_features(token)

            if features["short"] and features["no_vowel"] and token.islower():
                teencode_like_tokens[token.casefold()] += 1
                row_has_suspicious = True

            if features["repeated"]:
                elongated_tokens[token] += 1
                row_has_suspicious = True

            if features["mixed_alnum"]:
                mixed_alnum_tokens[token] += 1
                row_has_suspicious = True

            if features["accentless"] and features["lowercase_ascii"] and len(token) >= 3:
                accentless_ascii_tokens[token] += 1
                row_has_suspicious = True

            if features["repeated"] or features["mixed_alnum"] or (
                features["accentless"] and features["lowercase_ascii"] and len(token) >= 3
            ):
                possible_misspellings[token] += 1

        if row_has_suspicious:
            rows_with_suspicious_vocab += 1

    token_stats = {
        "total_tokens": int(sum(token_counter.values())),
        "unique_tokens": int(len(token_counter)),
    }

    return {
        "text_column": text_column,
        "token_stats": token_stats,
        "rows_with_suspicious_vocab": rows_with_suspicious_vocab,
        "rows_with_suspicious_vocab_ratio": percentage(rows_with_suspicious_vocab, len(series)),
        "teencode_like_tokens": _counter_to_list(teencode_like_tokens),
        "elongated_tokens": _counter_to_list(elongated_tokens),
        "accentless_ascii_tokens": _counter_to_list(accentless_ascii_tokens),
        "mixed_alnum_tokens": _counter_to_list(mixed_alnum_tokens),
        "possible_misspellings": _counter_to_list(possible_misspellings),
    }
