from fastapi import APIRouter

router = APIRouter()

@router.post("/generate-full-exam")
async def generate_full_exam():
    return {
        "message": "통합 워크플로우 정상 작동 중!",
        "process": "RAG -> 중요도 분석 -> 문제 생성 -> 평가"
    }