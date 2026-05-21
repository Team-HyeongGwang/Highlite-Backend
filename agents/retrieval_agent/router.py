import shutil
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pathlib import Path
from db.database import get_db

from .schemas import RAGPipelineRequest
from .service import run_pdf_pipeline

router = APIRouter()

@router.post("/test-retrieval")
async def test_retrieval():
    return {"message": "RAG 에이전트 라우터 정상 작동 중!"}

@router.post("/upload-pdf")
async def upload_and_process_pdf(
    file: UploadFile = File(..., description="처리할 PDF 파일을 업로드하세요"),
    user_id: int = Form(..., description="사용자 ID"),
    group_id: str = Form(..., description="그룹 ID"),
    db: AsyncSession = Depends(get_db)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드할 수 있습니다.")

    upload_dir = Path("temp_uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_path = upload_dir / file.filename
    try:
        with pdf_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"\n[Router] 파일 임시 저장 성공: {pdf_path}")

        # 🔥 이름이 변경된 마스터 파이프라인 가동!
        await run_pdf_pipeline(
            pdf_path=pdf_path,
            user_id=user_id,
            group_id=group_id,
            session=db
        )
        
        return {
            "status": "success", 
            "message": f"'{file.filename}' 파일이 마스터 파이프라인을 완벽하게 통과했습니다! 🎯"
        }

    except Exception as e:
        print(f"[Router Error] 파이프라인 중단됨: {e}")
        raise HTTPException(status_code=500, detail=f"파이프라인 처리 중 에러 발생: {str(e)}")
        
    finally:
        if pdf_path.exists():
            pdf_path.unlink()
            print(f"[Router] 임시 파일 청소 완료: {pdf_path}")