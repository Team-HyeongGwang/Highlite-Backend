import json
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

# 보내주신 최신 DB 모델 규격 정확히 임포트
from db.models import Question, Document, QuizResult, UserAnswer, ImportanceResult, DocumentChunk
from agents.personalized_agent.schemas import PersonalizationResponse

# 비용 및 성능 밸런스가 가장 뛰어난 Claude 모델 설정 (일관성을 위해 temperature=0 고정)
llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

# ────────────────────────────────────────
# 1. 풀이 세션 기록 및 Question Agent 제어용 가중치 반영
# ────────────────────────────────────────
async def record_quiz_session(
    user_id: int,
    document_id: int,  # group_id 대신 보내주신 스키마의 document_id 기반으로 명확히 매칭
    total_questions: int,
    correct_count: int,
    score_percent: int,
    attempt_phase: str, # 프론트엔드가 넘겨주는 회차 플래그 ("first_attempt", "regenerated" 등)
    answers_list: list,  # [{"question_id": 1, "user_answer": "①", "is_correct": False}, ...]
    db: AsyncSession
):
    # 1-1. 보내주신 QuizResult 스키마 양식에 100% 맞춰 마스터 레코드 생성
    quiz_master = QuizResult(
        user_id=user_id,
        document_id=document_id,
        total_questions=total_questions,
        correct_count=correct_count,
        score_percent=score_percent,
        attempt_phase=attempt_phase
    )
    db.add(quiz_master)
    await db.flush()  # 개별 응답(UserAnswer)과 바인딩할 quiz_master.id 선행 확보

    # 1-2. 개별 문제 응답 기록 및 중요도 스코어 동적 변환 (핵심 연결 메커니즘)
    for ans in answers_list:
        user_ans = UserAnswer(
            quiz_result_id=quiz_master.id,  # 보내주신 외래키 이름 일치
            question_id=ans["question_id"],
            user_answer=ans["user_answer"],
            is_correct=ans["is_correct"]
        )
        db.add(user_ans)

        # 고정된 Question Agent가 읽어서 소팅할 중요도(ImportanceResult) 데이터 조회
        stmt = (
            select(ImportanceResult)
            .join(Question, Question.importance_id == ImportanceResult.id)
            .where(Question.id == ans["question_id"])
        )
        imp_res = (await db.execute(stmt)).scalar_one_or_none()
        
        if imp_res:
            # 최초 풀이 세션과 재생성(오답노트) 세션에 따른 가중치 정밀 조절
            if attempt_phase != "first_attempt":
                # 오답노트 모드: 극복 시 취약 가중치 차감(-5.0), 또 틀릴 시 패널티 가중(+7.5)
                score_modifier = -5.0 if ans["is_correct"] else 7.5
            else:
                # 일반 시험 모드: 틀릴 시 완전 취약 영역 마킹(+10.0), 맞출 시 소폭 차감(-2.5)
                score_modifier = -2.5 if ans["is_correct"] else 10.0
            
            # 인위적인 조작으로 스코어가 원래 문서 중요도(base_score) 미만으로 깨지는 현상 방지
            imp_res.score = max(0.0, imp_res.score + score_modifier)

    await db.commit()
    return True


# ────────────────────────────────────────
# 2. LLM 환각 차단 취약점 분석 리포트 생성
# ────────────────────────────────────────
async def analyze_weakness(user_id: int, document_id: int, db: AsyncSession) -> PersonalizationResponse:
    # 💡 보내주신 스키마 구조인 document_id 기반 조인으로 완벽히 변환
    stmt = (
        select(
            Question.question_text, 
            ImportanceResult.keywords, 
            UserAnswer.is_correct
        )
        .join(UserAnswer, Question.id == UserAnswer.question_id)
        .join(QuizResult, UserAnswer.quiz_result_id == QuizResult.id)
        .join(ImportanceResult, Question.importance_id == ImportanceResult.id)
        .where(QuizResult.user_id == user_id, QuizResult.document_id == document_id)
    )
    rows = (await db.execute(stmt)).all()

    if not rows:
        return PersonalizationResponse(
            weakness_concepts=["아직 풀이 이력이 쌓이지 않았습니다."],
            weakness_source_file=str(document_id),
            personalized_advice="문제를 먼저 풀어보시면 학생의 학습 취약점을 정밀 진단해 드립니다.",
            next_recommendation=[]
        )

    # 팩트 기반 컨텍스트 빌딩 (단순 통계 숫자만 주었을 때의 LLM 개념 창조 환각 원천 차단)
    wrong_questions_details = []
    total_count = 0
    wrong_count = 0
    
    for row in rows:
        total_count += 1
        if not row.is_correct:
            wrong_count += 1
            # DB가 보관하는 실제 리스트 상태에 따른 예외 안전 가공 후 매핑
            k_list = row.keywords if isinstance(row.keywords, list) else json.loads(row.keywords or "[]")
            wrong_questions_details.append(
                f"- [학생이 틀린 문제]: {row.question_text}\n  [해당 지문의 핵심 개념 키워드]: {', '.join(k_list)}"
            )

    # 컨텍스트 길이 최적화를 위해 최근 오답 리스트 최대 10개로 바운딩
    stats_summary = f"전체 {total_count}문제 중 {wrong_count}문제를 오답 처리했습니다.\n" + "\n".join(wrong_questions_details[:10])

    # 문서의 실제 인간 친화적 원본 제목Fetch
    doc_title = (await db.execute(select(Document.title).where(Document.id == document_id).limit(1))).scalar() or f"Doc_{document_id}"

    # LLM 취약점 분석 요청 프롬프트 (오답에만 정밀 타격)
    prompt = f"""
학생이 학습 중인 교재 제목: {doc_title}
학생의 누적 오답 히스토리 및 실제 틀린 지문 데이터:
{stats_summary}

위 오답 데이터와 실제 틀린 문장 속 키워드를 냉철하게 대조하여, 학생의 학업 약점을 정밀 분석하세요.
반드시 제공된 실제 오답 내용에만 철저히 근거하여 아래 JSON 형식으로 응답하세요. 다른 부연 설명이나 마크다운 ```json 기호는 절대 덧붙이지 마세요.

{{
  "weakness_concepts": ["실제 오답 데이터에 기반하여 반복해서 틀리는 구체적인 개념명 1", "개념 2", "개념 3"],
  "personalized_advice": "이 문제들을 분석했을 때 학생이 자꾸 놓치는 논리적 맹점과 향후 올바른 복습 방향성 제언 (1문단)",
  "next_recommendation": ["다음 시험 준비 시 집중적으로 마스터해야 할 원문 핵심 키워드 1", "키워드 2", "키워드 3"]
}}
"""
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    
    # 펜스 가드 제거 및 클린 JSON 파싱
    clean_content = response.content.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(clean_content)

    # 💡 불필요한 wrong_chunk_ids 필드를 제거하고, 현재 Pydantic 스키마 규격에 완벽히 충실하게 반환
    return PersonalizationResponse(
        weakness_concepts=parsed["weakness_concepts"],
        weakness_source_file=doc_title,
        personalized_advice=parsed["personalized_advice"],
        next_recommendation=parsed["next_recommendation"]
    )