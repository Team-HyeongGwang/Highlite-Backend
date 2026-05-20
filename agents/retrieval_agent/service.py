
from typing import List
import os
from agents.importance_agent.schemas import ImportanceRequest
from dotenv import load_dotenv

from langchain_core.runnables import RunnableLambda
from langchain_openai import OpenAIEmbeddings
from sqlalchemy.ext.asyncio import AsyncSession
from db.models import DocumentChunk
from agents.pdf_pipeline_agent.schemas import PDFChunk
from agents.importance_agent.service import analyze_chunk_importance
from common.schemas import VisualCue

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small", # 임베딩 모델 고민 필요
    api_key=OPENAI_API_KEY,
)

# ── PDFChunk 받아오기 ────────────────────────────────────────
async def receive_chunks(input: dict) -> dict:
    chunks = input["chunks"]
    print(f"[rag_agent] 받은 청크 수: {len(chunks)}")
    for chunk in chunks:
        print(f"  - page={chunk.page} para={chunk.paragraph_index} content={chunk.content[:30]}...")
    return input


# ── 임베딩 계산 ────────────────────────────────────────
async def embed_chunks(input: dict) -> dict:
    chunks: list[PDFChunk] = input["chunks"]
    texts = [chunk.content for chunk in chunks]

    embeddings = await embeddings_model.aembed_documents(texts)

    for chunk, emb in zip(chunks, embeddings):
        chunk.embedding = emb
    return input


# ── 임베딩 값 DB에 저장 ────────────────────────────────────────
async def save_embeddings_to_db(input: dict) -> dict:
    chunks: list[PDFChunk] = input["chunks"]
    document_id: int = input["document_id"]
    session: AsyncSession = input["session"]

    for chunk in chunks:
        db_chunk = DocumentChunk(
            document_id=document_id,
            page_number=chunk.page,
            original_text=chunk.content,
            embedding=chunk.embedding,
            meta_data={
                "pdf_name": chunk.pdf_name,
                "paragraph_index": chunk.paragraph_index,
                "handwriting_color": chunk.handwriting_color,
                "highlight_color": chunk.highlight_color,
            },
        )
        
        session.add(db_chunk)
        await session.flush() 
        chunk.db_id = db_chunk.id  
        
    return input
# ── importance_agent에게 값 전달 ──────────────────────────────────────────
async def send_to_importance_agent(input: dict) -> dict:
    chunks: list[PDFChunk] = input["chunks"]
    session: AsyncSession = input["session"]

    # TODO: 추후 input dict에서 받아오도록 변경
    group_id = "default_group"
    doc_type = "pdf"
    highlighter_ranking = {"yellow": 1, "green": 2, "pink": 3}
    pen_ranking = {"red": 1, "blue": 2, "black": 3}

    for chunk in chunks:
        visual_cues: list[VisualCue] = []

        if chunk.highlight_color:
            visual_cues.append(VisualCue(
                type="highlight",
                color=chunk.highlight_color,
                target_text=chunk.content,  # ✅ chunk.content로 대통합!
            ))

        if chunk.handwriting_color:
            visual_cues.append(VisualCue(
                type="pen",
                color=chunk.handwriting_color,
                target_text=chunk.content,
            ))

        request = ImportanceRequest(
            group_id=group_id,
            chunk_id=chunk.db_id,
            doc_type=doc_type,
            original_text=chunk.content,
            meta_data=visual_cues,
            highlighter_ranking=highlighter_ranking,
            pen_ranking=pen_ranking,
        )

        print(f"\n[Sender: rag_chain] importance_agent로 데이터를 보냅니다.")
        print(f"  - Chunk ID: {request.chunk_id}")
        print(f"  - Visual Cues: {request.meta_data}")
        print(f"  - Original Text: {request.original_text[:30]}...")

        await analyze_chunk_importance(request, session)
        
    return input


# ── 체인 조립 ──────────────────────────────────────────
rag_chain = (
    RunnableLambda(receive_chunks)
    | RunnableLambda(embed_chunks)
    | RunnableLambda(save_embeddings_to_db)
    | RunnableLambda(send_to_importance_agent)
)
