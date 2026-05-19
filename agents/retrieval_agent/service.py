
# List[ChunkExtraction] 받아서
# 임베딩 후 벡터DB 저장

from typing import List
import os
from dotenv import load_dotenv

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from sqlalchemy.ext.asyncio import AsyncSession
from db.models import DocumentChunk
from agents.pdf_pipeline_agent.schemas import PDFChunk

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
                "has_handwriting": chunk.annotation.has_handwriting,
                "has_highlight": chunk.annotation.has_highlight,
                "handwriting_content": chunk.annotation.handwriting_content,
                "highlight_content": chunk.annotation.highlight_content,
            },
        )
        session.add(db_chunk)
    await session.commit()
    return input

# ── 체인 조립 ──────────────────────────────────────────
rag_chain = (
    RunnableLambda(receive_chunks)
    | RunnableLambda(embed_chunks)
    | RunnableLambda(save_embeddings_to_db)
)
