import shutil
import uuid
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
    db: AsyncSession = Depends(get_db)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드할 수 있습니다.")

    upload_dir = Path("temp_uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_path = upload_dir / file.filename
    group_id = uuid.uuid4() 

    try:
        with pdf_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        document_ids = await run_pdf_pipeline( 
            pdf_path=pdf_path,
            user_id=user_id,
            group_id=group_id,
            session=db
        )

        return {
            "status": "success",
            "group_id": str(group_id),  # uuid 직렬화 에러 방지
            "code": 200,
            "documents": [{"document_id": str(doc_id)} for doc_id in document_ids],  # uuid 직렬화 에러 방지
            "message": f"'{file.filename}' : Documents uploaded and saved successfully"
        }

    except Exception as e:
        print(f"[Router Error] 파이프라인 중단됨: {e}")
        return {
            "status": "error",
            "code": 500,
            "message": "Failed to save document"
        }

    finally:
        if pdf_path.exists():
            pdf_path.unlink()
            print(f"[Router] 임시 파일 청소 완료: {pdf_path}")