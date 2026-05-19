
# test_run.py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import asyncio
from agents.pdf_pipeline_agent.service import extract_from_pdf, run_pdf_pipeline
from agents.pdf_pipeline_agent.schemas import parse_chunk
from agents.retrieval_agent.service import rag_chain

from db.database import AsyncSessionLocal  # 프로젝트 DB 세션 팩토리


async def main():
    pdf_path = Path("C:/Users/jyeo0ng/Downloads/ALGOS_intermidiate_week3_modified.pdf")

    # ── 1. PDF → PDFChunk ──────────────────────────────
    print("=== 1. PDF 파싱 ===")
    pdf_name = pdf_path.stem
    raw_chunks = extract_from_pdf(pdf_path)
    chunks = [parse_chunk(raw, pdf_name) for raw in raw_chunks]
    print(f"파싱된 청크 수: {len(chunks)}")
    for chunk in chunks[:3]:
        print(f"  - page={chunk.page} para={chunk.paragraph_index} content={chunk.content[:40]}...")

    # ── 2. 임베딩 확인 ─────────────────────────────────
    print("\n=== 2. 임베딩 + DB 저장 ===")
    async with AsyncSessionLocal() as session:
        result = await rag_chain.ainvoke({
            "chunks": chunks,
            "document_id": 1,  # 테스트용 document_id
            "session": session,
        })

    # ── 3. 결과 확인 ───────────────────────────────────
    print("\n=== 3. 결과 확인 ===")
    saved_chunks = result["chunks"]
    print(f"총 청크 수: {len(saved_chunks)}")
    for chunk in saved_chunks[:3]:
        print(f"\npage={chunk.page} para={chunk.paragraph_index}")
        print(f"content: {chunk.content[:50]}...")
        print(f"embedding 길이: {len(chunk.embedding) if chunk.embedding else 'None'}")


asyncio.run(main())