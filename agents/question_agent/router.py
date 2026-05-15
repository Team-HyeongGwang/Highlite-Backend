from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_db
from agents.question_agent.schemas import (
    QuestionGenerateRequest,
    QuestionGenerateResponse,
    RegenerateRequest,
    RegenerateResponse,
)
from agents.question_agent.service import (
    generate_questions_service,
    regenerate_question_service,
)

router = APIRouter()

# ────────────────────────────────────────
# 1. 문제 생성
# ────────────────────────────────────────
@router.post("/generate", response_model=QuestionGenerateResponse)
async def generate_questions(
    request: QuestionGenerateRequest,
    db: AsyncSession = Depends(get_db),
):
    try:
        return await generate_questions_service(request, db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# ────────────────────────────────────────
# 2. 피드백 기반 재생성
# ────────────────────────────────────────
@router.post("/regenerate", response_model=RegenerateResponse)
async def regenerate_question(
    request: RegenerateRequest,
    db: AsyncSession = Depends(get_db),
):
    if request.retry_count >= 3:
        raise HTTPException(status_code=400, detail="재생성은 최대 3회까지만 가능합니다.")

    try:
        return await regenerate_question_service(request, db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))