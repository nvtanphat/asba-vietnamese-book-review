"""ABSA inference endpoint — a quick way to exercise the model on its own."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.auth import get_current_user
from app.db.models import AnalyzedReview, User
from app.db.session import get_db
from app.schemas import AbsaResult, AnalyzeIn
from app.services import absa_service

router = APIRouter(prefix="/absa", tags=["absa"])


@router.post("/analyze", response_model=AbsaResult)
def analyze(
    payload: AnalyzeIn,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> AbsaResult:
    result = absa_service.analyze(payload.text)

    # Persisted so week/month trend charts (see /reviews/history/summary) survive a
    # browser refresh or a different device logging in — the client-side session
    # history in localStorage only covers the current browser session.
    db.add(
        AnalyzedReview(
            shop_id=user.shop_id,
            text=result["text"],
            overall=result["overall"],
            overall_probs=result["overall_probs"],
            aspects=result["aspects"],
        )
    )
    db.commit()

    return AbsaResult(**result)
