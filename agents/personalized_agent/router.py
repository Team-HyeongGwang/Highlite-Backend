from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from uuid import UUID

from db.database import get_db
from agents.personalized_agent.schemas import (
    BatchSubmissionRequest,
    DocumentQuizResults,
    WrongAnswerItem,
)
from agents.personalized_agent.service import (
    record_quiz_session,
    get_quiz_results,
    get_wrong_answers,
)

router = APIRouter()

# ────────────────────────────────────────
# 테스트용 엔드포인트
# ────────────────────────────────────────
@router.post("/test-personalization")
async def test_personalization():
    return {"message": "개인화 에이전트 라우터 정상 작동 중!"}


# ────────────────────────────────────────
# 1. 배치 제출 엔드포인트
# ────────────────────────────────────────
@router.post("/submit-session", response_model=bool)
async def submit_quiz_session(
    req: BatchSubmissionRequest,
    db: AsyncSession = Depends(get_db)
):
    try:
        total = len(req.answers)
        correct = sum(1 for a in req.answers if a.is_correct)
        score_percent = int(correct / total * 100) if total > 0 else 0

        return await record_quiz_session(
            user_id=req.user_id,
            group_id=req.group_id,
            total_questions=total,
            correct_count=correct,
            score_percent=score_percent,
            attempt_phase=req.attempt_phase,
            answers_list=[
                {
                    "question_id": a.question_id,
                    "user_answer": a.user_answer,
                    "is_correct": a.is_correct,
                }
                for a in req.answers
            ],
            db=db,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"세션 저장 에러: {str(e)}")


# ────────────────────────────────────────
# 2. 문서별 퀴즈 결과 목록 조회
# ────────────────────────────────────────
@router.get("/quiz-results/{user_id}/{group_id}", response_model=List[DocumentQuizResults])
async def get_quiz_results_endpoint(
    user_id: int,
    group_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    try:
        return await get_quiz_results(user_id=user_id, group_id=group_id, db=db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"조회 에러: {str(e)}")


# ────────────────────────────────────────
# 3. 특정 회차 오답 목록 조회
# ────────────────────────────────────────
@router.get("/wrong-answers/{quiz_result_id}", response_model=List[WrongAnswerItem])
async def get_wrong_answers_endpoint(
    quiz_result_id: int,
    db: AsyncSession = Depends(get_db)
):
    try:
        return await get_wrong_answers(quiz_result_id=quiz_result_id, db=db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"조회 에러: {str(e)}")
