from fastapi import APIRouter

router = APIRouter()

@router.post("/test-retrieval")
async def test_retrieval():
    return {"message": "RAG 검색 에이전트 라우터 정상 작동 중!"}