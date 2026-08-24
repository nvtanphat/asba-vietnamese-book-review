"""Review sampling + persisted analysis-history endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.auth import get_current_user
from app.db.models import AnalyzedReview, User
from app.db.session import get_db
from app.schemas import AnalyzedReviewOut, TikiSampleOut
from app.services.tiki_fetch import TikiFetchError, fetch_live_negative_review

router = APIRouter(prefix="/reviews", tags=["reviews"])

# Matches absa_service._ASPECT_LABELS's keys — the six fixed aspects the model scores.
ASPECT_COLS = ["as_content", "as_physical", "as_price", "as_packaging", "as_delivery", "as_service"]


@router.get("/tiki-sample", response_model=TikiSampleOut)
def tiki_sample(user: User = Depends(get_current_user)) -> dict:
    """Fetch one real, live review from Tiki to prefill the ABSA analyzer."""
    try:
        return fetch_live_negative_review()
    except TikiFetchError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc


@router.get("/history", response_model=list[AnalyzedReviewOut])
def list_history(
    limit: int = Query(default=200, le=1000),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[AnalyzedReview]:
    """Most-recent-first log of every /absa/analyze call for the current shop."""
    stmt = (
        select(AnalyzedReview)
        .where(AnalyzedReview.shop_id == user.shop_id)
        .order_by(AnalyzedReview.created_at.desc())
        .limit(limit)
    )
    return list(db.scalars(stmt).all())


@router.get("/history/summary")
def history_summary(
    groupby: str = Query(default="week", pattern="^(week|month)$"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    """Weekly/monthly aggregates for the /stats dashboard's long-run trend charts.

    Bucketed in Python rather than SQL (date_trunc/strftime) so this works identically
    against SQLite (dev default) and Postgres (infra/docker-compose.yml) without a
    dialect-specific query — review volume for a single shop is small enough that
    fetching the rows and grouping in-process is simpler than the portability tradeoff.
    """
    stmt = (
        select(AnalyzedReview)
        .where(AnalyzedReview.shop_id == user.shop_id)
        .order_by(AnalyzedReview.created_at)
    )
    rows = list(db.scalars(stmt).all())

    buckets: dict[str, dict] = {}
    for row in rows:
        if groupby == "week":
            iso_year, iso_week, _ = row.created_at.isocalendar()
            period = f"{iso_year}-W{iso_week:02d}"
        else:
            period = row.created_at.strftime("%Y-%m")

        bucket = buckets.setdefault(
            period,
            {
                "period": period,
                "total": 0,
                "positive": 0,
                "neutral": 0,
                "negative": 0,
                "aspect_negative": {col: 0 for col in ASPECT_COLS},
            },
        )
        bucket["total"] += 1
        bucket[row.overall] = bucket.get(row.overall, 0) + 1
        for aspect in row.aspects or []:
            if aspect.get("sentiment") == "negative" and aspect.get("aspect") in bucket["aspect_negative"]:
                bucket["aspect_negative"][aspect["aspect"]] += 1

    return {"groupby": groupby, "buckets": sorted(buckets.values(), key=lambda b: b["period"])}
