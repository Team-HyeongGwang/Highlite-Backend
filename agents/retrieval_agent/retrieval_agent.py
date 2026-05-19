
# List[ChunkExtraction] 받아서
# 임베딩 후 벡터DB 저장
# BatchRetrievalResponse 반환

from typing import TypedDict, List, Optional, Any
import os
import heapq
import asyncio

from langgraph.graph import StateGraph, START, END

from sqlalchemy.ext.asyncio import AsyncSession
from db.crud_document_chunk import get_chunks_by_document, get_document_chunk_by_id
from db.models import DocumentChunk

import openai

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small" # 임베딩 모델 고민 필요
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")

openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, OPENAI_API_KEY)


# ── 임베딩 계산 ────────────────────────────────────────
async def compute_embedding(text: str) -> List[float]:
    """단일 텍스트 임베딩 계산"""
    response = await openai_client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


async def compute_embeddings_batch(
    texts: List[str],
    batch_size: int = 100,
) -> List[List[float]]:
    """배치 임베딩 계산 (OpenAI는 한 번에 최대 2048개 지원)"""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = await openai_client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=batch,
        )
        # 순서 보장: response.data는 index 기준 정렬됨
        sorted_data = sorted(response.data, key=lambda x: x.index)
        all_embeddings.extend([item.embedding for item in sorted_data])

    return all_embeddings