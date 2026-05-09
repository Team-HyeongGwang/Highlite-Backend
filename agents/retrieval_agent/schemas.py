from pydantic import BaseModel, Field
from typing import List

from common.schemas import VisualCue

# PDF에서 추출된 청크 데이터
class ChunkExtraction(BaseModel):
    page_number: int
    original_text: str
    meta_data: List[VisualCue] = Field(default_factory=list) 

# 파일 1개에 대한 기본 정보
class DocumentMetadata(BaseModel):
    title: str = Field(..., description="파일 이름")
    doc_type: str = Field("textbook", description="'textbook' 또는 'summary_note'")

# 여러 개의 파일을 한 번에 업로드할 때의 요청 
class BatchUploadRequest(BaseModel):
    user_id: int
    group_id: str = Field(..., description="같이 업로드된 파일들을 하나로 묶는 세트 ID (예: 'set-123')")
    documents_info: List[DocumentMetadata] = Field(..., description="업로드할 파일 정보 리스트")

# 다중 문서 처리 완료 응답
class BatchRetrievalResponse(BaseModel):
    group_id: str = Field(..., description="처리 완료된 세트 ID")
    processed_documents_count: int = Field(..., description="성공적으로 처리된 문서 개수")
    total_chunks_count: int = Field(..., description="DB에 저장된 총 청크 개수")
    message: str = "다중 문서 분석 및 벡터 DB 저장 완료"