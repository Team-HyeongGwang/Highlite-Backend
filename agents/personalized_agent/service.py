from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import UUID

from db.models import Question, Document, DocumentChunk, QuizResult, UserAnswer, ImportanceResult


def _score_to_imp(score: float) -> str:
    if score >= 7: return "R"
    if score >= 4: return "O"
    return "Y"

def _type_to_kor(qt: str) -> str:
    return {"multiple_choice": "객관식", "ox": "OX", "fill_in_the_blank": "빈칸채우기"}.get(qt, "객관식")

# ────────────────────────────────────────
# 1. 풀이 세션 전체 기록 및 스코어 가중 드라이빙
# ────────────────────────────────────────
async def record_quiz_session(
    user_id: int,
    group_id: UUID,
    total_questions: int,
    correct_count: int,
    score_percent: int,
    attempt_phase: str,
    answers_list: list,
    db: AsyncSession
):
    doc_result = await db.execute(
        select(Document.id).where(Document.group_id == group_id).limit(1)
    )
    document_id = doc_result.scalar()
    if not document_id:
        raise ValueError(f"group_id '{group_id}'에 해당하는 문서를 찾을 수 없습니다.")

    quiz_master = QuizResult(
        user_id=user_id,
        document_id=document_id,
        total_questions=total_questions,
        correct_count=correct_count,
        score_percent=score_percent,
        attempt_phase=attempt_phase
    )
    db.add(quiz_master)
    await db.flush()

    for ans in answers_list:
        user_ans = UserAnswer(
            quiz_result_id=quiz_master.id,
            question_id=ans["question_id"],
            user_answer=ans["user_answer"],
            is_correct=ans["is_correct"]
        )
        db.add(user_ans)

        stmt = (
            select(ImportanceResult)
            .join(Question, Question.importance_id == ImportanceResult.id)
            .where(Question.id == ans["question_id"])
        )
        imp_res = (await db.execute(stmt)).scalar_one_or_none()

        if imp_res:
            if attempt_phase != "first_attempt":
                score_modifier = -2.5 if ans["is_correct"] else 3.0
            else:
                score_modifier = -1.5 if ans["is_correct"] else 5.0

            imp_res.score = max(0.0, imp_res.score + score_modifier)

    await db.commit()
    return True


# ────────────────────────────────────────
# 2. 문서별 퀴즈 결과 목록 조회
# ────────────────────────────────────────
async def get_quiz_results(user_id: int, group_id: UUID, db: AsyncSession) -> list:
    doc_stmt = select(Document).where(Document.group_id == group_id)
    documents = (await db.execute(doc_stmt)).scalars().all()

    grouped = []
    for doc in documents:
        quiz_stmt = (
            select(QuizResult)
            .where(QuizResult.user_id == user_id, QuizResult.document_id == doc.id)
            .order_by(QuizResult.created_at)
        )
        quiz_results = (await db.execute(quiz_stmt)).scalars().all()

        attempts = []
        for idx, qr in enumerate(quiz_results):
            attempts.append({
                "id": qr.id,
                "round": idx + 1,
                "date": qr.created_at.strftime("%Y-%m-%d %H:%M"),
                "total": qr.total_questions,
                "correct": qr.correct_count,
                "wrong": qr.total_questions - qr.correct_count,
            })

        if attempts:
            grouped.append({
                "id": str(doc.id),
                "title": doc.title,
                "total_count": len(attempts),
                "attempts": attempts,
            })

    return grouped


# ────────────────────────────────────────
# 3. 특정 회차의 오답 목록 조회
# ────────────────────────────────────────
async def get_wrong_answers(quiz_result_id: int, db: AsyncSession) -> list:
    stmt = (
        select(
            UserAnswer.user_answer,
            Question.question_type,
            Question.question_text,
            Question.options,
            Question.answer,
            Question.explanation,
            ImportanceResult.score,
            DocumentChunk.page_number,
        )
        .join(Question, Question.id == UserAnswer.question_id)
        .join(ImportanceResult, ImportanceResult.id == Question.importance_id)
        .join(DocumentChunk, DocumentChunk.id == ImportanceResult.chunk_id)
        .where(UserAnswer.quiz_result_id == quiz_result_id)
        .where(UserAnswer.is_correct == False)
        .order_by(UserAnswer.id)
    )
    rows = (await db.execute(stmt)).all()

    result = []
    for idx, row in enumerate(rows):
        options_list = None
        if row.options:
            options_list = [f"{k} {v}" for k, v in row.options.items()]

        result.append({
            "id": f"Q{str(idx + 1).zfill(2)}",
            "imp": _score_to_imp(row.score),
            "type": _type_to_kor(row.question_type),
            "text": row.question_text,
            "options": options_list,
            "my_ans": row.user_answer or "",
            "correct": row.answer,
            "source": f"p.{row.page_number}",
            "exp": row.explanation,
        })

    return result
