from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from db.database import get_db
from .schemas import RankingRequest
from .service import create_ranking, get_ranking

router = APIRouter()

@router.post("/test-ranking")
async def test_retrieval():
    return {"message": "Ranking 입력 라우터 정상 작동 중!"}

@router.post("/color-rank")
async def create_rank(
    request: RankingRequest,
    user_id: int,  # 나중에 JWT로 교체
    db: AsyncSession = Depends(get_db)
):
    await create_ranking(user_id, request, db)
    return {"message": "랭킹 저장 완료"}


@router.get("/color-rank")
async def get_rank(
    user_id: int,
    db: AsyncSession = Depends(get_db)
):
    try:
        result = await get_ranking(user_id, db)
        
        if result is None:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "code": 404,
                    "data": {"message": "Ranking not found"}
                }
            )
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "code": 200,
                "data": {
                    "user_id": user_id,
                    "highlighter_ranking": result["highlighter_ranking"],
                    "pen_ranking": result["pen_ranking"]
                }
            }
        )

    except Exception:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "code": 500,
                "data": {"message": "Internal server error"}
            }
        )