from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import update, select
from db.models import User
from .schemas import RankingRequest

# 사용자로부터 형광펜 및 펜 중요도 순위를 받아와서 db에 저장
async def create_ranking(
    user_id: int,
    request: RankingRequest,
    db: AsyncSession
):
    values = {}
    if request.highlighter_ranking is not None:
        values["highlighter_ranking"] = request.highlighter_ranking
    if request.pen_ranking is not None:
        values["pen_ranking"] = request.pen_ranking

    if not values:
        return

    await db.execute(
        update(User)
        .where(User.id == user_id)
        .values(**values)
    )
    await db.commit()

# 사용자에게 형광펜 및 펜 중요도 순위를 반환
async def get_ranking(
    user_id: int,
    db: AsyncSession
):
    result = await db.execute(
        select(User.highlighter_ranking, User.pen_ranking)
        .where(User.id == user_id)
    )
    row = result.first()
    
    if row is None:
        return None
        
    return {
        "highlighter_ranking": row.highlighter_ranking,
        "pen_ranking": row.pen_ranking
    }