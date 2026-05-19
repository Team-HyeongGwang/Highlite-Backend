from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from .schemas import ImportanceRequest, ImportanceResponse
from .service import analyze_chunk_importance
from db.database import get_db

router = APIRouter()

@router.post("/", response_model=ImportanceResponse)
async def analyze_importance(
    request: ImportanceRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    [1타 강사 AI] PDF 텍스트와 시각 정보를 분석하여 중요도 점수를 계산합니다.
    """
    result = await analyze_chunk_importance(request, db)
    return result