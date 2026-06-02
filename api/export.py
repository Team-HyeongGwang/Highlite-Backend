from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import fitz  # PyMuPDF
import openai
from dotenv import load_dotenv

from db.database import get_db
from db.models import ImportanceResult, DocumentChunk, Document

load_dotenv()

router = APIRouter()
gpt_client = openai.AsyncOpenAI()


@router.get("/summary")
async def export_summary(
    group_id: str = Query(..., description="문서 그룹 ID"),
    format: str = Query("md", description="내보내기 형식: md 또는 pdf"),
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(ImportanceResult, DocumentChunk, Document)
        .join(DocumentChunk, ImportanceResult.chunk_id == DocumentChunk.id)
        .join(Document, DocumentChunk.document_id == Document.id)
        .where(Document.group_id == group_id)
        .order_by(DocumentChunk.page_number, ImportanceResult.score.desc())
    )
    rows = (await db.execute(stmt)).all()

    if not rows:
        raise HTTPException(status_code=404, detail="해당 문서의 중요도 분석 결과가 없습니다.")

    title = rows[0][2].title
    safe_title = title.replace(" ", "_")

    synthesized = await _synthesize_summary(title, rows)

    if format == "md":
        content = _build_markdown(title, synthesized)
        return Response(
            content=content.encode("utf-8"),
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{safe_title}_summary.md"'},
        )
    elif format == "pdf":
        pdf_bytes = _build_pdf(title, synthesized)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{safe_title}_summary.pdf"'},
        )
    else:
        raise HTTPException(status_code=400, detail="format은 'md' 또는 'pdf'만 지원합니다.")


async def _synthesize_summary(title: str, rows) -> str:
    chunks_data = []
    for importance, chunk, _ in rows:
        if importance.score < 4.0:
            continue
        entry = f"[{chunk.page_number}페이지 / 중요도 {importance.score:.1f}]\n"
        if importance.summary:
            entry += f"내용: {importance.summary}\n"
        if importance.keywords:
            entry += f"키워드: {', '.join(importance.keywords)}\n"
        chunks_data.append(entry)

    if not chunks_data:
        return "요약할 내용이 없습니다."

    context = "\n".join(chunks_data)

    prompt = f"""당신은 대학생 및 수험생을 위한 학습 요약 노트를 작성하는 전문 튜터입니다.
아래는 '{title}' 교재를 AI가 분석한 중요 개념 데이터입니다.
이 데이터를 바탕으로 학습자가 실제로 읽고 공부할 수 있는 체계적인 요약 노트를 작성해주세요.

[작성 규칙]
- 중요도 높은 개념을 중심으로 핵심 내용을 서술형으로 설명
- 단순히 키워드 나열이 아닌, 개념 간의 연관성과 맥락을 포함
- 섹션을 나눌 때는 "## 개념명" 형식 사용
- 각 개념 아래에 3~5문장으로 명확하게 설명
- 마지막에 "## 핵심 키워드 정리" 섹션으로 전체 키워드를 한 줄씩 정리
- 중요도 점수나 페이지 번호는 본문에 노출하지 말 것

[분석 데이터]
{context}

위 데이터를 참고하여 학습 요약 노트를 한국어로 작성해주세요."""

    try:
        response = await gpt_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception:
        raise HTTPException(status_code=502, detail="요약 생성 중 AI 서버 오류가 발생했습니다.")


def _build_markdown(title: str, synthesized_text: str) -> str:
    lines = [f"# {title} 요약본", ""]
    lines += synthesized_text.splitlines()
    lines.append("")
    return "\n".join(lines)


def _build_pdf(title: str, synthesized_text: str) -> bytes:
    W, H, MARGIN = 595, 842, 50
    doc = fitz.open()

    def new_page():
        return doc.new_page(width=W, height=H), MARGIN

    page, y = new_page()

    def writebox(text: str, fontsize: int, rect_h: int, gap: int, indent: int = 0):
        nonlocal page, y
        if y + rect_h + gap > H - MARGIN:
            page, y = new_page()
        rect = fitz.Rect(MARGIN + indent, y, W - MARGIN, y + rect_h)
        page.insert_textbox(rect, text, fontname="korea", fontsize=fontsize, align=0)
        y += rect_h + gap

    def est_h(text: str, fontsize: int, indent: int = 0) -> int:
        chars_per_line = max(1, int((W - 2 * MARGIN - indent) / (fontsize * 0.6)))
        lines = max(1, (len(text) + chars_per_line - 1) // chars_per_line)
        return lines * int(fontsize * 1.5) + 6

    writebox(f"{title} 요약본", fontsize=16, rect_h=28, gap=20)

    for line in synthesized_text.splitlines():
        stripped = line.strip()
        if not stripped:
            y += 8
            continue
        if stripped.startswith("## "):
            writebox(stripped[3:], fontsize=13, rect_h=24, gap=10)
        elif stripped.startswith("# "):
            writebox(stripped[2:], fontsize=15, rect_h=26, gap=12)
        elif stripped.startswith("- "):
            writebox(stripped, fontsize=10, rect_h=est_h(stripped, 10, 15), gap=4, indent=15)
        else:
            writebox(stripped, fontsize=10, rect_h=est_h(stripped, 10, 10), gap=6, indent=10)

    return doc.tobytes()
