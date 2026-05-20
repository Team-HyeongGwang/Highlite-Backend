# test_run.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import asyncio
from db.database import AsyncSessionLocal
from agents.pdf_pipeline_agent.test_service import create_pdf_chunks

async def main():
    pdf_path = Path("C:/Users/jyeo0ng/Downloads/데종분테스트.pdf")

    async with AsyncSessionLocal() as session:
        await create_pdf_chunks(
            pdf_path=pdf_path,
            user_id=1,        # 테스트용 (users 테이블에 있는 id)
            group_id="test-group",
            session=session,
        )
    
    print("완료! Supabase에서 documents, document_chunks 테이블 확인해보세요.")

asyncio.run(main())