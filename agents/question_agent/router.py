from fastapi import APIRouter, Depends, HTTPException
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_db
from agents.question_agent.schemas import (
    QuestionGenerateRequest,
    QuestionGenerateResponse,
    QuestionsByGroupResponse,
    WrongAnswersResponse,  # ← 추가
    RegenerateRequest,
    RegenerateResponse,
    SubmitAnswerRequest,
    SubmitAnswerResponse,
    RegenerateFromWrongRequest,
    QuestionListRequest,
    QuestionListResponse,
    DeleteQuizResultRequest,
    DeleteQuizResultResponse,
)
from agents.question_agent.service import (
    generate_questions_service,
    get_questions_by_group_service,
    get_wrong_answers_service,  # ← 추가
    regenerate_question_service,
    submit_answers_service,
    regenerate_from_wrong_service,
    get_question_list_service,
    delete_quiz_results_service,
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

# ────────────────────────────────────────
# 3. 채점
# ────────────────────────────────────────
@router.post("/submit", response_model=SubmitAnswerResponse)
async def submit_answers(
    request: SubmitAnswerRequest,
    db: AsyncSession = Depends(get_db),
):
    try:
        return await submit_answers_service(request, db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# ────────────────────────────────────────
# 4. 오답 기반 재생성
# ────────────────────────────────────────
@router.post("/regenerate-from-wrong", response_model=QuestionGenerateResponse)
async def regenerate_from_wrong(
    request: RegenerateFromWrongRequest,
    db: AsyncSession = Depends(get_db),
):
    try:
        return await regenerate_from_wrong_service(request, db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# ────────────────────────────────────────
# 5. 문서별 생성 문제 리스트 조회
# ────────────────────────────────────────
@router.get("/list", response_model=QuestionListResponse)
async def get_question_list(
    user_id: int,
    document_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    try:
        request = QuestionListRequest(user_id=user_id, document_id=document_id)
        return await get_question_list_service(request, db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# ────────────────────────────────────────
# 6. 회차 삭제
# ────────────────────────────────────────
@router.delete("/quiz-result", response_model=DeleteQuizResultResponse)
async def delete_quiz_results(
    request: DeleteQuizResultRequest,
    db: AsyncSession = Depends(get_db),
):
    try:
        return await delete_quiz_results_service(request, db)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# ────────────────────────────────────────
# 7. quiz_group_id로 문제 조회
# ────────────────────────────────────────
@router.get("/questions-by-group", response_model=QuestionsByGroupResponse)
async def get_questions_by_group(
    quiz_group_id: str,
    db: AsyncSession = Depends(get_db),
):
    try:
        return await get_questions_by_group_service(quiz_group_id, db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# ────────────────────────────────────────
# 8. 오답 조회
# ────────────────────────────────────────
@router.get("/wrong-answers", response_model=WrongAnswersResponse)
async def get_wrong_answers(
    quiz_result_id: int,
    db: AsyncSession = Depends(get_db),
):
    try:
        return await get_wrong_answers_service(quiz_result_id, db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))