from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_db
from agents.personalized_agent.schemas import AnswerSubmissionRequest
from agents.personalized_agent.service import record_quiz_session

router = APIRouter()

# ────────────────────────────────────────
# 테스트용 엔드포인트
# ────────────────────────────────────────
@router.post("/test-personalization")
async def test_personalization():
    return {"message": "개인화 에이전트 라우터 정상 작동 중!"}


# ────────────────────────────────────────
# 1. 낱개형 데이터를 받아서 묶음형 서비스로 변환해 주는 엔드포인트
# ────────────────────────────────────────
@router.post("/submit-session", response_model=bool)
async def submit_quiz_session(
    req: AnswerSubmissionRequest, 
    db: AsyncSession = Depends(get_db)
):
    try:
        # 스웨거가 보내주는 낱개 데이터(req.group_id 등)를 
        # 서비스 함수(record_quiz_session) 규격에 맞게 리스트([ ]) 형태로 포장합니다.
        
        # 임시 안전 장치: document_id는 테스트용으로 1을 부여합니다.
        return await record_quiz_session(
            user_id=req.user_id,
            group_id=req.group_id,
            total_questions=1,
            correct_count=1 if req.is_correct else 0,
            score_percent=100 if req.is_correct else 0,
            attempt_phase=req.attempt_phase,
            answers_list=[{
                "question_id": req.question_id,
                "user_answer": req.user_answer,
                "is_correct": req.is_correct
            }],
            db=db
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"세션 저장 에러: {str(e)}")


