from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_db
from agents.evaluation_agent.schemas import (
    QuestionReviewRequest, QuestionReviewResponse,
    FeedbackRequest, FeedbackResponse,
)
from agents.evaluation_agent.agent import review, regenerate_from_feedback

router = APIRouter()

@router.post("/test-evaluation")
async def test_evaluation():
    return {"message": "평가 에이전트 라우터 정상 작동 중!"}


@router.post("/review", response_model=QuestionReviewResponse)
async def review_question(req: QuestionReviewRequest):
    return await review(req)


@router.post("/feedback", response_model=FeedbackResponse)
async def handle_feedback(req: FeedbackRequest, db: AsyncSession = Depends(get_db)):
    result = await regenerate_from_feedback(req.question_id, req.feedback_type, db)
    if result is None:
        raise HTTPException(status_code=404, detail="문제를 찾을 수 없습니다")
    return result
