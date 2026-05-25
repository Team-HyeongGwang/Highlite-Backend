from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from db.database import get_db
from .schemas import RankingRequest
from .service import create_ranking

router = APIRouter()

@router.post("/test-ranking")
async def test_retrieval():
    return {"message": "Ranking 입력 라우터 정상 작동 중!"}

@router.post("/rank")
async def create_rank(
    request: RankingRequest,
    user_id: int,  # 나중에 JWT로 교체
    db: AsyncSession = Depends(get_db)
):
    await create_ranking(user_id, request, db)
    return {"message": "랭킹 저장 완료"}