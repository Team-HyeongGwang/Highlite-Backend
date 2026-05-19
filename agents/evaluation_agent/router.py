from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_db
from agents.evaluation_agent.schemas import (
    QuestionReviewRequest, QuestionReviewResponse,
    FeedbackRequest, FeedbackResponse,
)
from agents.evaluation_agent.service import review, regenerate_from_feedback

router = APIRouter()

@router.post("/test-evaluation")
async def test_evaluation():
    return {"message": "평가 에이전트 라우터 정상 작동 중!"}


@router.post("/review", response_model=QuestionReviewResponse)
async def review_question(req: QuestionReviewRequest):
    return await review(req)


