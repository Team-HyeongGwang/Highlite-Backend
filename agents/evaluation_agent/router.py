from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_db
from agents.evaluation_agent.schemas import (
    QuestionReviewRequest, 
    QuestionReviewResponse,
)
# 💡 에러를 냈던 FeedbackRequest 무리를 완벽히 청소하고 쓸 것만 남겼습니다.
from agents.evaluation_agent.service import review, regenerate_from_feedback

router = APIRouter(
    prefix="/evaluation",
    tags=["Evaluation (검수 에이전트)"]
)

@router.post("/test-evaluation")
async def test_evaluation():
    return {"message": "평가 에이전트 라우터 정상 작동 중!"}


@router.post("/review", response_model=QuestionReviewResponse)
async def review_question(
    req: QuestionReviewRequest,
    db: AsyncSession = Depends(get_db) # 💡 안정적인 파이프라인 연동을 위해 db 세션 주입 추가
):
    """
    Question Agent가 생성한 문제를 받아 원문 기반으로 팩트 체크 및 완성도를 검수합니다.
    검수 결과(승인 여부, 점수, 피드백, 수정 제안 보기)를 규격화된 JSON으로 반환합니다.
    """
    try:
        return await review(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"문제 검수 중 서버 에러 발생: {str(e)}")