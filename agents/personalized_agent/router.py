from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_db
from agents.personalized_agent.schemas import AnswerSubmissionRequest, PersonalizationResponse

# 💡 폴더명과 파일명이 겹쳐 발생한 ImportError를 방지하기 위해 경로를 명확히 명시하고 함수를 직접 가져옵니다.
from agents.personalized_agent.personalized_agent import record_quiz_session, analyze_weakness

router = APIRouter()

# ────────────────────────────────────────
# 테스트용 엔드포인트
# ────────────────────────────────────────
@router.post("/test-personalization")
async def test_personalization():
    return {"message": "개인화 에이전트 라우터 정상 작동 중!"}


# ────────────────────────────────────────
# 1. 풀이 데이터 세션 통합 기록 엔드포인트
# ────────────────────────────────────────
@router.post("/submit-session", response_model=bool)
async def submit_quiz_session(
    req: AnswerSubmissionRequest, 
    db: AsyncSession = Depends(get_db)
):
    try:
        # 임포트 방식 변경에 맞춰 바로 함수를 호출합니다.
        return await record_quiz_session(
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


# ────────────────────────────────────────
# 2. 취약점 리포트 조회 엔드포인트
# ────────────────────────────────────────
@router.get("/weakness/{user_id}/{group_id}", response_model=PersonalizationResponse)
async def get_weakness_report(
    user_id: int, 
    group_id: str,
    db: AsyncSession = Depends(get_db)
):
    try:
        # 임포트 방식 변경에 맞춰 바로 함수를 호출합니다.
        return await analyze_weakness(
            user_id=user_id, 
            group_id=group_id, 
            db=db
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"리포트 생성 에러: {str(e)}")