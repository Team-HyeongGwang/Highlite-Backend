from fastapi import APIRouter
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from db.database import get_db

from .schemas import RAGPipelineRequest
from .service import rag_chain

router = APIRouter()

@router.post("/test-retrieval")
async def test_retrieval():
    return {"message": "RAG 검색 에이전트 라우터 정상 작동 중!"}

@router.post("/rag-pipeline", response_model=dict)
async def run_rag_pipeline(
    request: RAGPipelineRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    [RAG Pipeline] PDF 청크를 받아 임베딩하고 DB에 저장한 뒤, 중요도 분석 에이전트로 전달
    """
    # 1. 체인이 기대하는 dict 형태로 데이터 구성 (FastAPI가 받아온 db 세션 주입)
    chain_input = {
        "chunks": request.chunks,
        "document_id": request.document_id,
        "session": db
    }
    
    try:
        # 2. 체인 실행! (이때 추가해두신 print 로그들이 터미널에 찍힙니다)
        await rag_chain.ainvoke(chain_input)
        
        return {"status": "success", "message": f"{len(request.chunks)}개의 청크 처리 완료"}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}