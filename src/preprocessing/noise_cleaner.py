from __future__ import annotations

import re
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\d[\s.-]?){8,12}\b")
HTML_HINT_RE = re.compile(r"<[^>]+>")


def _to_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def strip_html(value: Any) -> str | None:
    text = _to_text(value)
    if text is None:
        return None
    if not HTML_HINT_RE.search(text):
        return text
    return BeautifulSoup(text, "html.parser").get_text(" ", strip=True)


def normalize_noise(value: Any, url_token: str = "__url__", email_token: str = "__email__", phone_token: str = "__phone__") -> str | None:
    text = strip_html(value)
    if text is None:
        return None
    text = URL_RE.sub(url_token, text)
    text = EMAIL_RE.sub(email_token, text)
    text = PHONE_RE.sub(phone_token, text)
    return text


def normalize_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_noise)
