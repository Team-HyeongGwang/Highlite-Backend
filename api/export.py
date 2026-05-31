from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import fitz  # PyMuPDF

from db.database import get_db
from db.models import ImportanceResult, DocumentChunk, Document

router = APIRouter()


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

    if format == "md":
        content = _build_markdown(title, rows)
        return Response(
            content=content.encode("utf-8"),
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{safe_title}_summary.md"'},
        )
    elif format == "pdf":
        pdf_bytes = _build_pdf(title, rows)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{safe_title}_summary.pdf"'},
        )
    else:
        raise HTTPException(status_code=400, detail="format은 'md' 또는 'pdf'만 지원합니다.")


def _build_markdown(title: str, rows) -> str:
    lines = [f"# {title} 요약본", ""]
    current_page = None

    for importance, chunk, _ in rows:
        if not importance.summary and not importance.keywords:
            continue

        if chunk.page_number != current_page:
            current_page = chunk.page_number
            lines += [f"## {current_page}페이지", ""]

        if importance.summary:
            lines += [importance.summary, ""]

        if importance.keywords:
            lines += [f"**핵심 키워드:** {' · '.join(importance.keywords)}", ""]

        lines += [f"*중요도 {importance.score:.1f}/10*", "", "---", ""]

    return "\n".join(lines)


def _build_pdf(title: str, rows) -> bytes:
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

    writebox(f"{title} 요약본", fontsize=16, rect_h=24, gap=16)

    current_page_num = None
    for importance, chunk, _ in rows:
        if not importance.summary and not importance.keywords:
            continue

        if chunk.page_number != current_page_num:
            current_page_num = chunk.page_number
            writebox(f"[ {current_page_num}페이지 ]", fontsize=12, rect_h=20, gap=8)

        if importance.summary:
            writebox(importance.summary, fontsize=10, rect_h=48, gap=4, indent=10)

        if importance.keywords:
            writebox("키워드: " + " · ".join(importance.keywords), fontsize=9, rect_h=16, gap=4, indent=10)

        writebox(f"중요도 {importance.score:.1f}/10", fontsize=8, rect_h=14, gap=12, indent=10)

    return doc.tobytes()
