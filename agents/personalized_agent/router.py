from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_db
from agents.personalized_agent.schemas import AnswerSubmissionRequest, PersonalizationResponse
from agents.personalized_agent import personalized_agent as service

router = APIRouter()

# ────────────────────────────────────────
# 1. 풀이 데이터 세션 통합 기록 엔드포인트
# ────────────────────────────────────────
@router.post("/submit-session", response_model=bool)
async def submit_quiz_session(
    req: AnswerSubmissionRequest, 
    db: AsyncSession = Depends(get_db)
):
    try:
        # 단일 파일 채점 결과 마스터 기록 처리
        return await service.record_quiz_session(
            user_id=req.user_id,
            document_id=req.document_id, 
            total_questions=req.total_questions,
            correct_count=req.correct_count,
            score_percent=req.score_percent,
            attempt_phase=req.attempt_phase,
            answers_list=req.answers_list,
            db=db
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"세션 저장 에러: {str(e)}")



@router.get("/weakness/{user_id}/{group_id}", response_model=PersonalizationResponse)
async def get_weakness_report(
    user_id: int, 
    group_id: str, 
    db: AsyncSession = Depends(get_db)
):
    try:
        return await service.analyze_weakness(
            user_id=user_id, 
            group_id=group_id, 
            db=db
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"리포트 생성 에러: {str(e)}")