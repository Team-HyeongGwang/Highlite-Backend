from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import update
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