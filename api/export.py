from fastapi import APIRouter, Depends, Query, HTTPException, Body
from fastapi.responses import Response
from pydantic import BaseModel
from urllib.parse import quote
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
import fitz  # PyMuPDF
import openai
from dotenv import load_dotenv

from db.database import get_db
from db.models import ImportanceResult, DocumentChunk, Document, Question

load_dotenv()

router = APIRouter()
gpt_client = openai.AsyncOpenAI()

_EMOJI_MAP = {
    "yellow": "🟡", "red": "🔴", "orange": "🟠",
    "green": "🟢", "blue": "🔵", "purple": "🟣", "black": "⚫",
}

_COLOR_MAP = {
    "yellow": (1.0, 0.85, 0.0),
    "red":    (0.9, 0.15, 0.15),
    "orange": (1.0, 0.55, 0.0),
    "green":  (0.1, 0.65, 0.1),
    "blue":   (0.1, 0.35, 0.9),
    "purple": (0.5, 0.1,  0.8),
    "black":  (0.1, 0.1,  0.1),
}


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
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(safe_title)}_summary.md"},
        )
    elif format == "pdf":
        pdf_bytes = _build_pdf(title, synthesized)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(safe_title)}_summary.pdf"},
        )
    else:
        raise HTTPException(status_code=400, detail="format은 'md' 또는 'pdf'만 지원합니다.")


class RenderSummaryRequest(BaseModel):
    title: str
    synthesized_text: str


@router.post("/render-summary-pdf")
async def render_summary_pdf(request: RenderSummaryRequest):
    safe_title = request.title.replace(" ", "_")
    pdf_bytes = _build_pdf(request.title, request.synthesized_text)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(safe_title)}_summary.pdf"},
    )


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
        color = None
        if chunk.meta_data:
            color = chunk.meta_data.get("highlight_color") or chunk.meta_data.get("handwriting_color")
        if color:
            entry += f"색상: {color}\n"
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
- 입력 데이터에 "색상" 필드가 있는 섹션은 "## 개념명" 앞에 "[COLOR:색상명]" 태그를 붙여주세요 (예: "[COLOR:yellow]## 개념명")
- 색상 정보가 없는 섹션은 태그 없이 "## 개념명" 그대로 작성

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
    for line in synthesized_text.splitlines():
        if line.startswith("[COLOR:") and "]" in line:
            end = line.index("]")
            color = line[7:end]
            rest = line[end + 1:]
            emoji = _EMOJI_MAP.get(color, "")
            lines.append(f"{emoji} {rest}".strip() if emoji else rest)
        else:
            lines.append(line)
    lines.append("")
    return "\n".join(lines)


def _build_pdf(title: str, synthesized_text: str) -> bytes:
    W, H, MARGIN = 595, 842, 50
    doc = fitz.open()

    def new_page():
        return doc.new_page(width=W, height=H), MARGIN

    page, y = new_page()

    def writebox(text: str, fontsize: int, gap: int, indent: int = 0, color: tuple = None):
        nonlocal page, y
        if H - MARGIN - y < fontsize * 2:
            page, y = new_page()
        available_h = H - MARGIN - y
        rect = fitz.Rect(MARGIN + indent, y, W - MARGIN, y + available_h)
        if color:
            result = page.insert_textbox(rect, text, fontname="korea", fontsize=fontsize, align=0, color=color)
        else:
            result = page.insert_textbox(rect, text, fontname="korea", fontsize=fontsize, align=0)
        used_h = available_h - max(0.0, result)
        if color:
            page.draw_line(
                fitz.Point(MARGIN + indent - 5, y),
                fitz.Point(MARGIN + indent - 5, y + used_h),
                color=color, width=3,
            )
        y += used_h + gap

    writebox(f"{title} 요약본", fontsize=16, gap=20)

    for line in synthesized_text.splitlines():
        stripped = line.strip()
        if not stripped:
            y += 8
            continue

        color_rgb = None
        if stripped.startswith("[COLOR:") and "]" in stripped:
            end = stripped.index("]")
            color_rgb = _COLOR_MAP.get(stripped[7:end])
            stripped = stripped[end + 1:].lstrip()

        if stripped.startswith("## "):
            writebox(stripped[3:], fontsize=13, gap=10, color=color_rgb)
        elif stripped.startswith("# "):
            writebox(stripped[2:], fontsize=15, gap=12)
        elif stripped.startswith("- "):
            writebox(stripped, fontsize=10, gap=4, indent=15)
        else:
            writebox(stripped, fontsize=10, gap=6, indent=10)

    return doc.tobytes()


@router.get("/questions")
async def export_questions(
    group_id: str = Query(..., description="문서 그룹 ID"),
    format: str = Query("md", description="내보내기 형식: md 또는 pdf"),
    db: AsyncSession = Depends(get_db),
):
    max_round_stmt = (
        select(func.max(Question.round_number))
        .join(ImportanceResult, Question.importance_id == ImportanceResult.id)
        .join(DocumentChunk, ImportanceResult.chunk_id == DocumentChunk.id)
        .join(Document, DocumentChunk.document_id == Document.id)
        .where(Document.group_id == group_id)
    )
    max_round = (await db.execute(max_round_stmt)).scalar()

    if max_round is None:
        raise HTTPException(status_code=404, detail="해당 문서의 문제가 없습니다.")

    stmt = (
        select(Question, Document)
        .join(ImportanceResult, Question.importance_id == ImportanceResult.id)
        .join(DocumentChunk, ImportanceResult.chunk_id == DocumentChunk.id)
        .join(Document, DocumentChunk.document_id == Document.id)
        .where(Document.group_id == group_id)
        .where(Question.round_number == max_round)
        .order_by(Question.id)
    )
    rows = (await db.execute(stmt)).all()

    if not rows:
        raise HTTPException(status_code=404, detail="해당 문서의 문제가 없습니다.")

    title = rows[0][1].title
    safe_title = title.replace(" ", "_")
    questions = [row[0] for row in rows]

    if format == "md":
        content = _build_questions_markdown(title, questions)
        return Response(
            content=content.encode("utf-8"),
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(safe_title)}_questions.md"},
        )
    elif format == "pdf":
        pdf_bytes = _build_questions_pdf(title, questions)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(safe_title)}_questions.pdf"},
        )
    else:
        raise HTTPException(status_code=400, detail="format은 'md' 또는 'pdf'만 지원합니다.")


def _build_questions_markdown(title: str, questions: list) -> str:
    lines = [f"# {title} 문제지", ""]

    for i, q in enumerate(questions, 1):
        header = f"## 문제 {i}"
        if q.question_type:
            header += f"  ·  {q.question_type}"
        if q.difficulty:
            header += f"  ·  난이도: {q.difficulty}"
        lines.append(header)
        lines.append("")
        lines.append(q.question_text)
        lines.append("")
        if q.options:
            for key, value in q.options.items():
                lines.append(f"{key} {value}")
            lines.append("")
        lines.append("---")
        lines.append("")

    lines += ["", f"# {title} 정답 및 해설", ""]

    for i, q in enumerate(questions, 1):
        lines.append(f"## 문제 {i}")
        lines.append("")
        lines.append(f"**정답:** {q.answer}")
        lines.append("")
        lines.append(f"**해설:** {q.explanation}")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def _build_questions_pdf(title: str, questions: list) -> bytes:
    W, H, MARGIN = 595, 842, 50
    doc = fitz.open()

    def new_page():
        return doc.new_page(width=W, height=H), MARGIN

    page, y = new_page()

    def writebox(text: str, fontsize: int, gap: int, indent: int = 0):
        nonlocal page, y
        if H - MARGIN - y < fontsize * 2:
            page, y = new_page()
        available_h = H - MARGIN - y
        rect = fitz.Rect(MARGIN + indent, y, W - MARGIN, y + available_h)
        result = page.insert_textbox(rect, text, fontname="korea", fontsize=fontsize, align=0)
        y += (available_h - max(0.0, result)) + gap

    # 문제지
    writebox(f"{title} 문제지", fontsize=16, gap=20)

    for i, q in enumerate(questions, 1):
        header = f"문제 {i}"
        if q.question_type:
            header += f"  ·  {q.question_type}"
        if q.difficulty:
            header += f"  ·  난이도: {q.difficulty}"
        writebox(header, fontsize=12, gap=8)
        writebox(q.question_text, fontsize=10, gap=6, indent=10)
        if q.options:
            for key, value in q.options.items():
                writebox(f"{key} {value}", fontsize=10, gap=3, indent=20)
        y += 10

    # 정답 및 해설 — 새 페이지
    page, y = new_page()
    writebox(f"{title} 정답 및 해설", fontsize=16, gap=20)

    for i, q in enumerate(questions, 1):
        writebox(f"문제 {i}", fontsize=12, gap=6)
        writebox(f"정답: {q.answer}", fontsize=10, gap=4, indent=10)
        writebox(f"해설: {q.explanation}", fontsize=10, gap=14, indent=10)

    return doc.tobytes()
