from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from db.models import Question, Document, QuizResult, UserAnswer, ImportanceResult

# ────────────────────────────────────────
# 1. 풀이 세션 전체 기록 및 스코어 가중 드라이빙
# ────────────────────────────────────────
async def record_quiz_session(
    user_id: int,
    group_id: str,
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

    # 1-1. 기획팀 양식 테이블에 완벽히 매칭
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

    # 1-2. 낱개 문제 정오답 기록 및 출제 정렬 스코어 조작
    for ans in answers_list:
        user_ans = UserAnswer(
            quiz_result_id=quiz_master.id, 
            question_id=ans["question_id"],
            user_answer=ans["user_answer"],
            is_correct=ans["is_correct"]
        )
        db.add(user_ans)

        # Question Agent가 소팅하는 중요도 테이블 조회
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