from fastapi import APIRouter

router = APIRouter()

@router.post("/pdf-pipeline")
async def pdf_pipeline():
    return {"message": "PDF 파이프라인 라우터 정상 작동 중!"}