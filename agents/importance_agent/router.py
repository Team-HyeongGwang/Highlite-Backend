from fastapi import APIRouter

router = APIRouter()

@router.post("/test-analyze")
async def test_importance():
    return {"message": "중요도 분석 라우터 정상 작동 중!"}