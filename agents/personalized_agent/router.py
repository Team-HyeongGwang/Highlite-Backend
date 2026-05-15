from fastapi import APIRouter

router = APIRouter()

@router.post("/test-personalization")
async def test_personalization():
    return {"message": "개인화 에이전트 라우터 정상 작동 중!"}