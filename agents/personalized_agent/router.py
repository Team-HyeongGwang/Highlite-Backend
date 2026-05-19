from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_db
from agents.personalized_agent.schemas import AnswerSubmissionRequest, PersonalizationResponse
from agents.personalized_agent.service import record_answer, compute_bias, analyze_weakness, _session_log

router = APIRouter()

@router.post("/test-personalization")
async def test_personalization():
    return {"message": "개인화 에이전트 라우터 정상 작동 중!"}

@router.post("/submit-answer")
async def submit_answer(req: AnswerSubmissionRequest, db: AsyncSession = Depends(get_db)):
    result = await record_answer(req.user_id, req.group_id, req.question_id, req.user_answer, req.is_correct, db)
    if result is None:
        raise HTTPException(status_code=404, detail="문제를 찾을 수 없습니다")
    return {"message": "기록 완료", "is_correct": req.is_correct}


@router.get("/next-bias/{user_id}/{group_id}")
async def get_next_bias(user_id: int, group_id: str):
    bias = compute_bias(user_id, group_id)
    return {"user_id": user_id, "group_id": group_id, "bias": bias}


@router.get("/weakness/{user_id}/{group_id}", response_model=PersonalizationResponse)
async def get_weakness(user_id: int, group_id: str, db: AsyncSession = Depends(get_db)):
    if not _session_log[user_id][group_id]:
        raise HTTPException(status_code=404, detail="풀이 이력이 없습니다")
    return await analyze_weakness(user_id, group_id, db)