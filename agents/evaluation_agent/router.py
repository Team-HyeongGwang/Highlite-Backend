from fastapi import APIRouter

router = APIRouter()

@router.post("/test-evaluation")
async def test_evaluation():
    return {"message": "평가 및 개인화 에이전트 라우터 정상 작동 중!"}