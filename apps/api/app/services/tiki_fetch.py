"""Fetch one real, live review from Tiki's public JSON API for demo purposes.

Reuses the same public, unauthenticated endpoint as `ml/scrapers/crawler.py`
(`tiki.vn/api/v2/reviews`) but only pulls a single review on demand instead of running the full
multi-threaded crawl. This is a demo-only helper: it lets the ABSA analyzer UI run on genuine,
freshly-fetched Tiki content instead of hardcoded sample text.

A demo order code is appended to the fetched content purely as illustrative flavor text - real
Tiki reviews never contain an order code. The appended part is clearly separate from the real
review text.
"""

from __future__ import annotations

import random
from typing import Any

import requests

# Book product IDs on Tiki confirmed (at the time of writing) to have 1-2 star reviews.
# Sourced from real negative reviews already present in this project's ABSA training dataset.
DEMO_PRODUCT_IDS = [480040, 2287449, 94173462, 3071853]

# Illustrative order codes appended to fetched review text for the demo UI.
DEMO_ORDER_CODES = ["DH10231", "DH10232", "DH10235"]

# Real Tiki reviewers sometimes tag a short, non-complaint comment (e.g. "Cực kì hài lòng")
# with a low star rating by mistake - a data-quality quirk this project's own label-QA work
# ran into repeatedly. Filtering for longer text strongly biases toward genuine complaints
# that actually read as negative, which makes for a much clearer demo.
MIN_REVIEW_LENGTH = 40

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://tiki.vn/",
}


class TikiFetchError(RuntimeError):
    """Raised when a live review could not be fetched from Tiki."""


def fetch_live_negative_review() -> dict[str, Any]:
    """Fetch one real 1-2 star review from Tiki right now, with a demo order code appended."""
    session = requests.Session()
    product_ids = random.sample(DEMO_PRODUCT_IDS, len(DEMO_PRODUCT_IDS))
    for product_id in product_ids:
        for stars in (1, 2):
            try:
                resp = session.get(
                    "https://tiki.vn/api/v2/reviews",
                    params={"product_id": product_id, "stars": stars, "limit": 20},
                    headers=_HEADERS,
                    timeout=10,
                )
            except requests.RequestException as exc:
                raise TikiFetchError(f"Không kết nối được tới Tiki: {exc}") from exc

            if resp.status_code != 200:
                continue

            candidates = []
            for item in resp.json().get("data", []):
                text = (item.get("content") or item.get("title") or "").strip()
                if len(text) >= MIN_REVIEW_LENGTH:
                    candidates.append(text)

            if candidates:
                original_text = random.choice(candidates)
                order_code = random.choice(DEMO_ORDER_CODES)
                return {
                    "original_text": original_text,
                    "text": f"{original_text} (Đơn hàng {order_code})",
                    "order_code": order_code,
                    "product_id": product_id,
                    "stars": stars,
                }

    raise TikiFetchError("Không tìm thấy review nào từ Tiki lúc này, thử lại sau.")
