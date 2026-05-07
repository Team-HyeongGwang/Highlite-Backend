from fastapi import APIRouter

router = APIRouter()

@router.post("/test-question")
async def test_question():
    return {"message": "문제 생성 에이전트 라우터 정상 작동 중!"}