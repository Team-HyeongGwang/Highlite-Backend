import random, json, math, asyncio, httpx, anthropic, openai, os
import uuid as uuid_lib
import re
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from dotenv import load_dotenv

load_dotenv()

from db.models import Question, ImportanceResult, DocumentChunk, Document, User
from agents.personalized_agent.service import record_quiz_session
from agents.question_agent.schemas import (
    DeleteQuizResultRequest,
    DeleteQuizResultResponse,
    QuestionGenerateRequest,
    QuestionGenerateResponse,
    QuestionItem,
    QuestionsByGroupResponse,
    RegenerateRequest,
    RegenerateResponse,
    SubmitAnswerRequest,
    SubmitAnswerResponse,
    AnswerResult,
    RegenerateFromWrongRequest,
    QuestionListRequest,
    QuestionListResponse,
    DocumentItem,
    AttemptItem,
)

claude_client = anthropic.AsyncAnthropic()
gpt_client = openai.AsyncOpenAI()

EVALUATION_URL = os.getenv("EVALUATION_URL", "http://127.0.0.1:8000/evaluation/review")

# ────────────────────────────────────────
# 순위 결정
# ────────────────────────────────────────
def get_priority(meta_data, highlighter_ranking: dict, pen_ranking: dict) -> int:
    best = 99

    # 형식 1: 리스트 형식 [{"type": "highlight", "color": "yellow"}, ...]
    if isinstance(meta_data, list):
        for cue in meta_data:
            if not isinstance(cue, dict):
                continue
            color = cue.get("color", "")
            cue_type = cue.get("type", "")
            if cue_type == "highlight":
                rank = highlighter_ranking.get(color, 99)
            elif cue_type == "pen":
                rank = pen_ranking.get(color, 99)
            else:
                continue
            best = min(best, rank)

    # 형식 2: 딕셔너리 형식 {"highlight_color": "yellow", "handwriting_color": "blue"}
    elif isinstance(meta_data, dict):
        highlight_color = meta_data.get("highlight_color")
        handwriting_color = meta_data.get("handwriting_color")

        if highlight_color:
            rank = highlighter_ranking.get(highlight_color, 99)
            best = min(best, rank)
        if handwriting_color:
            rank = pen_ranking.get(handwriting_color, 99)
            best = min(best, rank)

    return best if best != 99 else 3

# ────────────────────────────────────────
# source_type 결정
# ────────────────────────────────────────
def get_source_type(meta_data) -> str:
    # 형식 1: 리스트 형식
    if isinstance(meta_data, list):
        for cue in meta_data:
            if not isinstance(cue, dict):
                continue
            if cue.get("type") == "pen":
                return "pen"
        return "highlight"

    # 형식 2: 딕셔너리 형식
    elif isinstance(meta_data, dict):
        if meta_data.get("handwriting_color"):
            return "pen"
        if meta_data.get("highlight_color"):
            return "highlight"

    return "highlight"

# ────────────────────────────────────────
# doc_type JSON 파싱
# ────────────────────────────────────────
def get_doc_category(doc_type_raw) -> str | None:
    if not doc_type_raw:
        return None
    try:
        parsed = json.loads(doc_type_raw) if isinstance(doc_type_raw, str) else doc_type_raw
        return parsed.get("type")  # "textbook" | "summary_note" | None
    except Exception:
        return None

# ────────────────────────────────────────
# 교재 유사 청크 탐색 (in-memory cosine similarity)
# ────────────────────────────────────────
def _cosine_similarity(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

def find_best_match_chunk(query_emb, textbook_chunks: list):
    if query_emb is None or not textbook_chunks:
        return None
    best, best_score = None, -1.0
    for chunk in textbook_chunks:
        if chunk.embedding is None:
            continue
        score = _cosine_similarity(list(query_emb), list(chunk.embedding))
        if score > best_score:
            best, best_score = chunk, score
    return best

# ────────────────────────────────────────
# 문제 유형 결정
# ────────────────────────────────────────
def get_question_type(priority: int) -> str:
    if priority == 1:
        return random.choice(["multiple_choice", "ox", "fill_in_the_blank"])
    else:
        return random.choice(["multiple_choice", "ox"])

# ────────────────────────────────────────
# 출제 수 분배 (1순위 50% / 2순위 30% / 3순위 20%)
# ────────────────────────────────────────
def distribute_counts(total: int, chunks_by_priority: dict) -> dict:
    ratio = {1: 0.5, 2: 0.3, 3: 0.2}
    counts = {}
    remainder = total

    for priority in [1, 2, 3]:
        chunks = chunks_by_priority.get(priority, [])
        if not chunks:
            continue
        counts[priority] = math.floor(total * ratio[priority])
        remainder -= counts[priority]

    for priority in [1, 2, 3]:
        if priority in counts and remainder > 0:
            counts[priority] += remainder
            remainder = 0
            break

    return counts

# ────────────────────────────────────────
# 프롬프트 생성
# ────────────────────────────────────────
def build_prompt(
    context_text: str,
    keywords: List[str],
    question_type: str,
    feedback_type: Optional[str] = None,
) -> str:
    type_guide = {
        "multiple_choice": """[4지선다 객관식 문제 조건]
- 정답은 반드시 논란의 여지가 없는 1개여야 합니다.
- 오답 선지(3개)는 아래의 '매력적 오답 제조 매커니즘' 중 적어도 2가지를 활용해 구성하세요:
  1. 원문의 원인과 결과를 뒤바꾼 선지
  2. 원문의 특정 키워드를 유사해 보이는 다른 오개념과 교묘하게 결합한 선지
  3. 지나친 일반화(예: 반드시, 항상, 전혀 등)를 포함하여 오류를 만든 선지
- 모든 선지는 문장 구조와 길이를 최대한 유사하게 맞춰 시각적 힌트를 배제하세요.
- options에 "①","②","③","④" 키로 보기 4개를 제공하세요.

[출제 포맷]
{{
  "question_text": "질문 본문만 작성 (①~④ 선지 내용이나 '문제:', 'Q.' 등의 접두어 포함 절대 엄금)",
  "options": {{"①": "선지1 내용", "②": "선지2 내용", "③": "선지3 내용", "④": "선지4 내용"}},
  "answer": "①", 
  "explanation": "최대 3문장 이내의 근거 중심 해설 (반드시 ①, ②, ③, ④ 기호를 명확히 언급하며 경어체로 작성)"
}}""",

        "ox": """[O/X 문제 조건]
- 문장은 단 한 줄로도 명확하게 참(O) 또는 거짓(X)이 판별되어야 합니다.
- 단순히 원문 문장에 '아니다'만 붙인 유치한 오답 문장은 금지합니다.
- 개념의 핵심 전제나 조건을 살짝 비틀어 깊은 이해를 요구하는 문장으로 구성하세요.
- options는 반드시 null로 설정하세요.

[출제 포맷]
{{
  "question_text": "참/거짓을 판별할 순수 문장 한 줄만 작성 ('다음 문장이 맞으면...' 같은 안내문구나 접두어 절대 엄금)",
  "options": null,
  "answer": "O",
  "explanation": "최대 3문장 이내의 근거 중심 해설 (경어체로 작성)"
}}""",

        "fill_in_the_blank": """[빈칸 채우기 문제 조건]
- 원문에서 가장 학술적으로 중요한 개념/단어 자리를 딱 1개만 '_____'로 표시하세요.
- 빈칸 바로 뒤에 붙는 조사(은/는, 이/가, 을/를)를 통해 정답의 힌트가 유추되지 않도록 문장을 매끄럽게 다듬으세요.
- options는 반드시 null로 설정하세요.
- **[핵심 금지]** question_text는 반드시 빈칸(_____)이 포함된 문장 자체여야 합니다. "다음 빈칸에 들어갈 용어를 쓰시오", "다음은 특정 과목에 대한 필기 내용이다. 빈칸에 들어갈 용어를 쓰시오." 같은 안내 문구로 시작하는 것은 절대 금지입니다.

[출제 포맷]
{{
  "question_text": "빈칸(_____)이 포함된 핵심 개념 문장 자체만 작성. 예시: '스택은 _____ 원칙을 따르는 자료구조이다.'",
  "options": null,
  "answer": "원문기준정답단어",
  "explanation": "최대 3문장 이내의 근거 중심 해설 (경어체로 작성)"
}}""",
    }[question_type]

    feedback_guide = ""
    if feedback_type:
        feedback_map = {
            "ambiguous":           "기존 문제가 애매하다는 피드백이 있었습니다. 정답이 복수 개가 되거나 해석의 여지가 갈리지 않도록 논리 구조를 더 명확하게 다듬으세요.",
            "wrong_answer":        "기존 문제의 정답이 틀렸다는 피드백이 있었습니다. 원문을 철저히 재검토하여 '원문 근거 기준'으로 완벽한 정답을 다시 지정하세요.",
            "unclear_explanation": "해설이 이해가 안 된다는 피드백이 있었습니다. 초등학생도 이해할 수 있을 만큼 원문의 근거 문장을 명확히 인용하며 쉽게 풀어쓰세요.",
            "irrelevant":          "문제가 내용과 관련 없다는 피드백이 있었습니다. 지엽적인 단어 장난이 아닌, 원문의 핵심 줄기와 거시적 맥락에 충실한 문제를 만드세요.",
        }
        feedback_guide = f"\n[중요 피드백 반영 지시]\n주의: {feedback_map.get(feedback_type, '')}"

    return f"""너는 출제 오류율 0%를 자랑하는 대학교 전공 시험 출제위원이야.
제공된 [원문]과 [핵심 키워드]를 바탕으로, 단순 암기(Recall)를 넘어 개념 적용 및 분석 능력을 평가할 수 있는 고품격 문제를 출제해줘.

[최우선 준수 원칙]
1. 실질 개념 출제 원칙: [원문]에 "[필기: ...]", "[별표]", "[밑줄]" 등의 메타 태그가 포함된 경우, 해당 태그 자체나 과목명·필기 형식을 묻는 문제는 절대 출제하지 마세요.
   추가 금지 패턴:
   - 강의명, 교수명, 수업 메타 정보를 묻는 문제
   - 원문 속 질문 문구를 그대로 문제로 전환하는 것
   - 선지가 '원문에서 언급됨', '원문에서 묘사됨' 형태로 원문 존재 여부만 묻는 문제
   - 원문에 포함된 ①②③④ 같은 번호 표시를 선지에 그대로 노출하는 것
   - '[필기: ...]' 태그의 내용을 빈칸으로 만드는 것
   반드시 원문의 핵심 개념(정의, 특징, 원리, 메커니즘)을 이해해야 풀 수 있는 문제를 출제하세요.
2. 단일 포인트 집중 출제: 제공된 [원문]에서 가장 핵심적인 개념 포인트 하나에만 집중하여 출제하세요. 여러 개념을 한 문제에 혼합하거나 지엽적인 세부 사항을 묻지 마세요.

[원문]
{context_text}

[핵심 키워드]
{', '.join(keywords)}

[출제 대원칙 - 필수 준수]
3. 엄격한 원문 근거주의: 오직 제공된 [원문]에 명시된 사실과 논리만을 바탕으로 출제하세요. 원문 외 지식이나 상식에 의존해야 풀 수 있는 문제는 절대 금지합니다.
4. 질문의 명확성: 문제 텍스트 자체만 읽어도 무엇을 묻는지 학습자가 한 번에 파악할 수 있어야 합니다.
5. 해설 퀄리티 및 길이 제한: 해설은 정답의 이유와 오답이 틀린 이유를 원문 근거를 바탕으로 최대 3문장 이내의 간결한 경어체('~입니다')로 작성하세요.
6. 강력한 포맷팅 제한: 모든 문제 유형의 question_text에 "문제:", "Q.", "보기:", "지문:" 등의 불필요한 접두어를 절대 포함하지 마세요. 객관식의 경우 question_text 내부에 ①~④ 선지를 절대 중복 기재하지 마세요.

{type_guide}{feedback_guide}

반드시 위 [출제 포맷] 구조를 완벽히 따른 순수 JSON 문자열만 출력하세요. Markdown 블록(```json)은 절대 포함하지 마세요.
"""

# ────────────────────────────────────────
# Claude 호출
# ────────────────────────────────────────
async def call_claude(prompt: str) -> dict:
    message = await claude_client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1000,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text
    clean = raw.replace("```json", "").replace("```", "").strip()
    result = json.loads(clean)
    # 접두어 방어 코드: "문제:" 등 제거
    if "question_text" in result:
        text = result["question_text"].strip()
        # 접두어 반복 제거 (공백/줄바꿈 포함)
        prefixes = ["문제:", "문제 :", "Q.", "Q .", "보기:", "지문:"]
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if text.lstrip().startswith(prefix):
                    text = text.lstrip()[len(prefix):].strip()
                    changed = True
        # 객관식 선지가 본문에 포함된 경우 제거
        has_real_options = (
            isinstance(result.get("options"), dict)
            and "①" in result.get("options", {})
        )
        if has_real_options and "①" in text:
            text = text[:text.index("①")].strip()
        result["question_text"] = text
    return result

# ────────────────────────────────────────
# GPT 호출
# ────────────────────────────────────────
async def call_gpt(prompt: str) -> dict:
    response = await gpt_client.chat.completions.create(
        model="gpt-4.1",
        max_tokens=1000,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content
    clean = raw.replace("```json", "").replace("```", "").strip()
    result = json.loads(clean)
    # 접두어 방어 코드: "문제:" 등 제거
    if "question_text" in result:
        text = result["question_text"].strip()
        # 접두어 반복 제거 (공백/줄바꿈 포함)
        prefixes = ["문제:", "문제 :", "Q.", "Q .", "보기:", "지문:"]
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if text.lstrip().startswith(prefix):
                    text = text.lstrip()[len(prefix):].strip()
                    changed = True
        # 객관식 선지가 본문에 포함된 경우 제거
        has_real_options = (
            isinstance(result.get("options"), dict)
            and "①" in result.get("options", {})
        )
        if has_real_options and "①" in text:
            text = text[:text.index("①")].strip()
        result["question_text"] = text
    return result

# ────────────────────────────────────────
# 평가 Agent 호출
# ────────────────────────────────────────
async def call_evaluation(
    group_id: str,
    source_chunk_text: str,
    question_text: str,
    options: Optional[dict],
    answer: str,
    explanation: str,
) -> dict:
    payload = {
        "group_id": group_id,
        "source_chunk_text": source_chunk_text,
        "question_text": question_text,
        "options": options,
        "answer": answer,
        "explanation": explanation,
    }
    async with httpx.AsyncClient(timeout=30) as http_client:
        response = await http_client.post(EVALUATION_URL, json=payload)
        response.raise_for_status()
        return response.json()

# ────────────────────────────────────────
# 공통 문제 생성 로직
# ────────────────────────────────────────
async def _generate_questions_from_chunks(
    group_id: str,
    chunks_by_priority: dict,
    question_count: int,
    db: AsyncSession,
    quiz_group_id: uuid_lib.UUID = None,
    round_number: int = None,
    textbook_chunks: list | None = None,
) -> List[QuestionItem]:
    counts = distribute_counts(question_count, chunks_by_priority)

    total_chunks = sum(len(v) for v in chunks_by_priority.values())
    print(f"\n[문제 생성] 🚀 문제 생성 파이프라인 시작 (목표: {question_count}문제 / 청크 수: {total_chunks}개)")
    print(f"[문제 생성] 우선순위 분배: {counts}")

    task_list = []
    used_chunk_ids = set()  # ← 중복 청크 방지용 집합

    for priority, target_count in counts.items():
        pool = chunks_by_priority.get(priority, [])
        if not pool:
            continue
        pool_sorted = sorted(pool, key=lambda x: x[0].score, reverse=True)

        # ── 수정: 같은 청크 중복 선택 방지 ──
        added = 0
        for importance, chunk in pool_sorted:
            if added >= target_count:
                break
            if chunk.id in used_chunk_ids:  # 이미 선택된 청크 스킵
                continue
            if not chunk.original_text or not chunk.original_text.strip():
                print(f"[문제 생성] chunk_id={chunk.id} original_text 없음 → 스킵")
                continue
            task_list.append((priority, importance, chunk))
            used_chunk_ids.add(chunk.id)
            added += 1

        # ── 수정: 해당 priority 청크가 부족하면 다른 priority 청크로 보충 ──
        if added < target_count:
            print(f"[문제 생성] priority={priority} 청크 부족 ({added}/{target_count}) → 다른 priority로 보충")
            for fallback_priority in [1, 2, 3]:
                if fallback_priority == priority:
                    continue
                fallback_pool = chunks_by_priority.get(fallback_priority, [])
                for importance, chunk in sorted(fallback_pool, key=lambda x: x[0].score, reverse=True):
                    if added >= target_count:
                        break
                    if chunk.id in used_chunk_ids:
                        continue
                    if not chunk.original_text or not chunk.original_text.strip():
                        continue
                    task_list.append((priority, importance, chunk))
                    used_chunk_ids.add(chunk.id)
                    added += 1
                if added >= target_count:
                    break

    print(f"[문제 생성] 총 {len(task_list)}개 청크 대상 선정 완료")

    async def generate_one(priority: int, importance, chunk, task_idx: int) -> Optional[tuple]:
        print(f"[문제 생성] [{task_idx+1}/{len(task_list)}] p.{chunk.page_number} 청크 → Claude + GPT 동시 출제 중...")

        question_type = get_question_type(priority)
        keywords = importance.keywords or []

        try:
            if textbook_chunks is not None and chunk.embedding is not None:
                best_textbook = find_best_match_chunk(chunk.embedding, textbook_chunks)
                if best_textbook:
                    print(f"[문제 생성] [{task_idx+1}] 교재 유사 청크 매칭 (p.{best_textbook.page_number})")
                    context_text = f"[필기본 내용]\n{chunk.original_text}\n\n[교재 관련 내용]\n{best_textbook.original_text}"
                else:
                    context_text = chunk.original_text
            else:
                context_text = chunk.original_text
            prompt = build_prompt(context_text, keywords, question_type)
        except Exception as e:
            print(f"[문제 생성] [{task_idx+1}] build_prompt 예외: {type(e).__name__}: {e}")
            return None

        if not prompt or not isinstance(prompt, str):
            print(f"[문제 생성] [{task_idx+1}] 프롬프트 생성 실패: prompt={prompt}")
            return None

        claude_result, gpt_result = await asyncio.gather(
            call_claude(prompt),
            call_gpt(prompt),
            return_exceptions=True,
        )

        type_label = {"multiple_choice": "객관식", "ox": "OX", "fill_in_the_blank": "빈칸채우기"}.get(question_type, question_type)
        claude_ok = not isinstance(claude_result, Exception)
        gpt_ok = not isinstance(gpt_result, Exception)
        print(f"[문제 생성] [{task_idx+1}/{len(task_list)}] 출제 완료 ({type_label}) | Claude: {'✅' if claude_ok else '❌'} / GPT: {'✅' if gpt_ok else '❌'}")

        valid_results = []
        eval_tasks = []
        for source, result in [("claude", claude_result), ("gpt", gpt_result)]:
            if isinstance(result, Exception):
                print(f"[문제 생성] [{task_idx+1}] {source} 실패: {type(result).__name__}: {result}")
                continue
            valid_results.append((source, result))
            eval_tasks.append(call_evaluation(
                group_id=group_id,
                source_chunk_text=chunk.original_text,
                question_text=result["question_text"],
                options=result.get("options"),
                answer=result["answer"],
                explanation=result["explanation"],
            ))

        if not eval_tasks:
            print(f"[평가 Agent] [{task_idx+1}/{len(task_list)}] ⚠️ 유효한 후보 없음 → 스킵")
            return None

        print(f"[평가 Agent] [{task_idx+1}/{len(task_list)}] {len(eval_tasks)}개 후보 품질 평가 중...")
        eval_results = await asyncio.gather(*eval_tasks, return_exceptions=True)

        candidates = []
        for (source, result), review in zip(valid_results, eval_results):
            if isinstance(review, Exception):
                continue
            candidates.append({"source": source, "result": result, "review": review})

        if not candidates:
            print(f"[평가 Agent] [{task_idx+1}/{len(task_list)}] ❌ 평가 실패 → 스킵")
            return None

        approved = [c for c in candidates if c["review"]["is_approved"]]
        if approved:
            best = max(approved, key=lambda x: x["review"]["quality_score"])
            print(f"[평가 Agent] [{task_idx+1}/{len(task_list)}] ✅ 승인 (score: {best['review']['quality_score']:.2f}, source: {best['source']})")
        else:
            revised = [c for c in candidates if c["review"].get("suggested_revision_text")]
            if revised:
                best = revised[0]
                best["result"]["question_text"] = best["review"]["suggested_revision_text"]
                if best["review"].get("suggested_revision_options"):
                    best["result"]["options"] = best["review"]["suggested_revision_options"]
                print(f"[평가 Agent] [{task_idx+1}/{len(task_list)}] 🔧 수정안 채택 (source: {best['source']})")
            else:
                best = max(candidates, key=lambda x: x["review"].get("quality_score", 0))
                print(f"[평가 Agent] [{task_idx+1}/{len(task_list)}] ⚠️ 미승인이지만 최고 점수 채택 (score: {best['review'].get('quality_score', 0):.2f})")

        return (priority, importance, chunk, question_type, best["result"])

    # ── 배치 병렬 실행 ──
    BATCH_SIZE = 10
    all_results = []
    total_batches = math.ceil(len(task_list) / BATCH_SIZE)

    for i in range(0, len(task_list), BATCH_SIZE):
        batch = task_list[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"\n[배치 처리] 📦 배치 {batch_num}/{total_batches} 시작 ({len(batch)}개 병렬 처리)")

        batch_results = await asyncio.gather(
            *[generate_one(p, imp, chunk, i + idx) for idx, (p, imp, chunk) in enumerate(batch)],
            return_exceptions=True,
        )
        all_results.extend(batch_results)

        success = sum(1 for r in batch_results if r and not isinstance(r, Exception))
        print(f"[배치 처리] ✅ 배치 {batch_num}/{total_batches} 완료 ({success}/{len(batch)} 성공)")

        if i + BATCH_SIZE < len(task_list):
            await asyncio.sleep(0.5)

    # ── DB 저장 ──
    generated: List[QuestionItem] = []
    print(f"\n[DB 저장] 💾 생성된 문제 DB 저장 시작...")

    for res in all_results:
        if res is None or isinstance(res, Exception):
            continue
        priority, importance, chunk, question_type, result = res

        question = Question(
            importance_id=importance.id,
            quiz_group_id=quiz_group_id,
            round_number=round_number,
            question_type=question_type,
            difficulty=str(priority),
            question_text=result["question_text"],
            options=result.get("options"),
            answer=result["answer"],
            explanation=result["explanation"],
        )
        db.add(question)
        await db.flush()

        source_type = get_source_type(chunk.meta_data or [])
        generated.append(QuestionItem(
            chunk_id=chunk.id,
            question_type=question_type,
            question_text=result["question_text"],
            options=result.get("options"),
            answer=result["answer"],
            explanation=result["explanation"],
            question_number=len(generated) + 1,
            priority=priority,
            source_type=source_type,
            page_number=chunk.page_number,
            question_id=question.id,
        ))

    print(f"[DB 저장] ✅ 완료 → 총 {len(generated)}문제 저장됨")
    print(f"[문제 생성] 🎉 파이프라인 종료\n")

    return generated

# ────────────────────────────────────────
# 1. 문제 생성 서비스
# ────────────────────────────────────────
async def generate_questions_service(
    request: QuestionGenerateRequest,
    db: AsyncSession,
) -> QuestionGenerateResponse:

    # 그룹 내 문서 목록 조회 → 듀얼 모드 감지
    docs_stmt = select(Document).where(Document.group_id == request.group_id)
    docs = (await db.execute(docs_stmt)).scalars().all()

    textbook_doc_ids = [d.id for d in docs if get_doc_category(d.doc_type) == "textbook"]
    notes_doc_ids = [d.id for d in docs if get_doc_category(d.doc_type) == "notes"]
    is_dual_mode = bool(textbook_doc_ids) and bool(notes_doc_ids)

    if is_dual_mode:
        print(f"[문제 생성] 듀얼 모드 감지 (교재 {len(textbook_doc_ids)}개 / 필기본 {len(notes_doc_ids)}개)")
        chunk_filter = DocumentChunk.document_id.in_(notes_doc_ids)
    else:
        chunk_filter = Document.group_id == request.group_id

    stmt = (
        select(ImportanceResult, DocumentChunk, Document)
        .join(DocumentChunk, ImportanceResult.chunk_id == DocumentChunk.id)
        .join(Document, DocumentChunk.document_id == Document.id)
        .where(Document.group_id == request.group_id)
        .where(chunk_filter)
    )
    rows = (await db.execute(stmt)).all()

    if not rows:
        raise ValueError("해당 문서의 중요도 분석 결과가 없습니다.")

    document = rows[0][2]
    user_q = await db.execute(select(User).where(User.id == document.user_id))
    user = user_q.scalar_one_or_none()

    highlighter_ranking = user.highlighter_ranking or {}
    pen_ranking = user.pen_ranking or {}

    # 듀얼 모드: 교재 chunk를 embedding 포함하여 미리 로드
    textbook_chunks = None
    if is_dual_mode:
        tb_stmt = (
            select(DocumentChunk)
            .where(DocumentChunk.document_id.in_(textbook_doc_ids))
            .where(DocumentChunk.embedding.isnot(None))
        )
        textbook_chunks = (await db.execute(tb_stmt)).scalars().all()
        print(f"[문제 생성] 교재 청크 {len(textbook_chunks)}개 로드 완료")

    chunks_by_priority = {1: [], 2: [], 3: []}
    for importance, chunk, _ in rows:
        priority = get_priority(
            chunk.meta_data or [],
            highlighter_ranking,
            pen_ranking,
        )
        chunks_by_priority[priority].append((importance, chunk))

    quiz_group_id = uuid_lib.uuid4()

    sibling_stmt = select(Document.id).where(Document.group_id == document.group_id)
    sibling_ids = [r[0] for r in (await db.execute(sibling_stmt)).all()]

    max_round_stmt = (
        select(func.max(Question.round_number))
        .join(ImportanceResult, Question.importance_id == ImportanceResult.id)
        .join(DocumentChunk, ImportanceResult.chunk_id == DocumentChunk.id)
        .where(DocumentChunk.document_id.in_(sibling_ids))
    )
    max_round = (await db.execute(max_round_stmt)).scalar() or 0
    next_round = max_round + 1

    generated = await _generate_questions_from_chunks(
        group_id=request.group_id,
        chunks_by_priority=chunks_by_priority,
        question_count=request.question_count,
        db=db,
        quiz_group_id=quiz_group_id,
        round_number=next_round,
        textbook_chunks=textbook_chunks,
    )

    await db.commit()
    return QuestionGenerateResponse(
        document_id=document.id,
        quiz_group_id=quiz_group_id,
        questions=generated
    )

# ────────────────────────────────────────
# 2. 피드백 기반 재생성 서비스
# ────────────────────────────────────────
async def regenerate_question_service(
    request: RegenerateRequest,
    db: AsyncSession,
) -> RegenerateResponse:

    prompt = build_prompt(
        request.context_text,
        request.keywords,
        request.question_type,
        feedback_type=request.feedback_type,
    )

    # Claude + GPT 병렬 호출
    claude_result, gpt_result = await asyncio.gather(
        call_claude(prompt),
        call_gpt(prompt),
        return_exceptions=True,
    )

    # evaluation agent 병렬 호출
    valid_results = []
    eval_tasks = []
    for source, result in [("claude", claude_result), ("gpt", gpt_result)]:
        if isinstance(result, Exception):
            print(f"[피드백 재생성] {source} 호출 실패: {result}")
            continue
        valid_results.append((source, result))
        eval_tasks.append(call_evaluation(
            group_id="feedback",  # 피드백 재생성은 group_id 불필요, 임시값
            source_chunk_text=request.context_text,
            question_text=result["question_text"],
            options=result.get("options"),
            answer=result["answer"],
            explanation=result["explanation"],
        ))

    if not eval_tasks:
        raise ValueError("Claude와 GPT 모두 문제 생성에 실패했습니다.")

    eval_results = await asyncio.gather(*eval_tasks, return_exceptions=True)

    candidates = []
    for (source, result), review in zip(valid_results, eval_results):
        if isinstance(review, Exception):
            print(f"[피드백 재생성] evaluation 실패: {review}")
            continue
        candidates.append({"source": source, "result": result, "review": review})

    if not candidates:
        raise ValueError("평가 Agent 호출에 실패했습니다.")

    # 최선 후보 선택
    approved = [c for c in candidates if c["review"]["is_approved"]]
    if approved:
        best = max(approved, key=lambda x: x["review"]["quality_score"])
        print(f"[피드백 재생성] ✅ 승인 (score: {best['review']['quality_score']:.2f}, source: {best['source']})")
    else:
        revised = [c for c in candidates if c["review"].get("suggested_revision_text")]
        if revised:
            best = revised[0]
            best["result"]["question_text"] = best["review"]["suggested_revision_text"]
            if best["review"].get("suggested_revision_options"):
                best["result"]["options"] = best["review"]["suggested_revision_options"]
            print(f"[피드백 재생성] 🔧 수정안 채택 (source: {best['source']})")
        else:
            best = max(candidates, key=lambda x: x["review"].get("quality_score", 0))
            print(f"[피드백 재생성] ⚠️ 미승인 최고 점수 채택 (score: {best['review'].get('quality_score', 0):.2f})")

    result = best["result"]

    # DB 업데이트
    q_result = await db.execute(select(Question).where(Question.id == request.question_id))
    question = q_result.scalar_one_or_none()

    if not question:
        raise ValueError("문제를 찾을 수 없습니다.")

    question.question_text = result["question_text"]
    question.options       = result.get("options")
    question.answer        = result["answer"]
    question.explanation   = result["explanation"]

    await db.commit()
    await db.refresh(question)

    return RegenerateResponse(
        question_id=question.id,
        question_type=request.question_type,
        **{k: result[k] for k in ["question_text", "options", "answer", "explanation"]},
    )

# ────────────────────────────────────────
# 3. 채점 + 저장
# ────────────────────────────────────────
async def submit_answers_service(
    request: SubmitAnswerRequest,
    db: AsyncSession,
) -> SubmitAnswerResponse:
    from db.models import UserAnswer, QuizResult

    results = []

    for item in request.answers:
        question_id = item["question_id"]
        submitted = item["submitted_answer"]

        q_result = await db.execute(
            select(Question).where(Question.id == question_id)
        )
        question = q_result.scalar_one_or_none()
        if not question:
            continue

        # 띄어쓰기 무시하고 채점
        def normalize(text: str) -> str:
            return re.sub(r'\s+', '', text.strip())

        is_correct = (normalize(submitted) == normalize(question.answer))
        results.append({
            "question_id": question_id,
            "submitted": submitted,
            "correct_answer": question.answer,
            "is_correct": is_correct,
            "explanation": question.explanation,
        })

    correct = sum(1 for r in results if r["is_correct"])
    total = len(results)
    score_percent = round((correct / total) * 100) if total > 0 else 0

    quiz_result = QuizResult(
        user_id=request.user_id,
        document_id=request.document_id,
        quiz_group_id=request.quiz_group_id,
        total_questions=total,
        correct_count=correct,
        score_percent=score_percent,
        attempt_phase=request.attempt_phase,
    )
    db.add(quiz_result)
    await db.flush()

    for r in results:
        answer = UserAnswer(
            quiz_result_id=quiz_result.id,
            question_id=r["question_id"],
            user_answer=r["submitted"],
            is_correct=r["is_correct"],
        )
        db.add(answer)

    await db.commit()

    # 채점 완료 후 ImportanceResult.score 개인화 업데이트
    try:
        doc_result = await db.execute(
            select(Document.group_id).where(Document.id == request.document_id)
        )
        group_id = doc_result.scalar_one_or_none()
        if group_id:
            answers_list = [
                {
                    "question_id": r["question_id"],
                    "user_answer": r["submitted"],
                    "is_correct": r["is_correct"],
                }
                for r in results
            ]
            await record_quiz_session(
                user_id=request.user_id,
                group_id=group_id,
                total_questions=total,
                correct_count=correct,
                score_percent=score_percent,
                attempt_phase=request.attempt_phase,
                answers_list=answers_list,
                db=db,
            )
    except Exception:
        pass

    return SubmitAnswerResponse(
        total=total,
        correct=correct,
        wrong=total - correct,
        results=[
            AnswerResult(
                question_id=r["question_id"],
                submitted_answer=r["submitted"],
                correct_answer=r["correct_answer"],
                is_correct=r["is_correct"],
                explanation=r["explanation"],
            )
            for r in results
        ],
    )

# ────────────────────────────────────────
# 4. 오답 기반 재생성
# ────────────────────────────────────────
async def regenerate_from_wrong_service(
    request: RegenerateFromWrongRequest,
    db: AsyncSession,
) -> QuestionGenerateResponse:
    from db.models import UserAnswer, QuizResult

    wrong_stmt = (
        select(UserAnswer.question_id)
        .join(QuizResult, UserAnswer.quiz_result_id == QuizResult.id)
        .where(QuizResult.user_id == request.user_id)
        .where(QuizResult.document_id == request.document_id)
        .where(UserAnswer.is_correct == False)
    )
    wrong_rows = (await db.execute(wrong_stmt)).all()
    wrong_question_ids = [r[0] for r in wrong_rows]

    wrong_chunk_ids = set()
    if wrong_question_ids:
        stmt = (
            select(DocumentChunk.id)
            .join(ImportanceResult, ImportanceResult.chunk_id == DocumentChunk.id)
            .join(Question, Question.importance_id == ImportanceResult.id)
            .where(Question.id.in_(wrong_question_ids))
        )
        wrong_chunk_rows = (await db.execute(stmt)).all()
        wrong_chunk_ids = {r[0] for r in wrong_chunk_rows}

    stmt = (
        select(ImportanceResult, DocumentChunk, Document)
        .join(DocumentChunk, ImportanceResult.chunk_id == DocumentChunk.id)
        .join(Document, DocumentChunk.document_id == Document.id)
        .where(Document.group_id == request.group_id)
    )
    rows = (await db.execute(stmt)).all()

    if not rows:
        raise ValueError("해당 문서의 중요도 분석 결과가 없습니다.")

    document = rows[0][2]
    user_q = await db.execute(select(User).where(User.id == document.user_id))
    user = user_q.scalar_one_or_none()
    highlighter_ranking = user.highlighter_ranking or {}
    pen_ranking = user.pen_ranking or {}

    chunks_by_priority = {1: [], 2: [], 3: []}
    for importance, chunk, _ in rows:
        if chunk.id in wrong_chunk_ids:
            priority = 1
        else:
            priority = get_priority(
                chunk.meta_data or [],
                highlighter_ranking,
                pen_ranking,
            )
        chunks_by_priority[priority].append((importance, chunk))

    quiz_group_id = uuid_lib.uuid4()

    sibling_stmt = select(Document.id).where(Document.group_id == document.group_id)
    sibling_ids = [r[0] for r in (await db.execute(sibling_stmt)).all()]

    max_round_stmt = (
        select(func.max(Question.round_number))
        .join(ImportanceResult, Question.importance_id == ImportanceResult.id)
        .join(DocumentChunk, ImportanceResult.chunk_id == DocumentChunk.id)
        .where(DocumentChunk.document_id.in_(sibling_ids))
    )
    max_round = (await db.execute(max_round_stmt)).scalar() or 0
    next_round = max_round + 1

    generated = await _generate_questions_from_chunks(
        group_id=request.group_id,
        chunks_by_priority=chunks_by_priority,
        question_count=request.question_count,
        db=db,
        quiz_group_id=quiz_group_id,
        round_number=next_round,
    )

    await db.commit()
    return QuestionGenerateResponse(
        document_id=document.id,
        quiz_group_id=quiz_group_id,
        questions=generated
    )

# ────────────────────────────────────────
# 5. 문서별 생성 문제 리스트 조회
# ────────────────────────────────────────
async def get_question_list_service(
    request: QuestionListRequest,
    db: AsyncSession,
) -> QuestionListResponse:
    from db.models import QuizResult

    doc_stmt = select(Document).where(Document.user_id == request.user_id)
    if request.document_id:
        doc_stmt = doc_stmt.where(Document.id == request.document_id)

    doc_rows = (await db.execute(doc_stmt)).scalars().all()

    if not doc_rows:
        return QuestionListResponse(documents=[])

    seen_group_ids = set()
    unique_docs = []
    for doc in doc_rows:
        g = str(doc.group_id)
        if g not in seen_group_ids:
            seen_group_ids.add(g)
            unique_docs.append(doc)

    documents = []
    for doc in unique_docs:
        sibling_stmt = select(Document.id).where(Document.group_id == doc.group_id)
        sibling_ids = [r[0] for r in (await db.execute(sibling_stmt)).all()]

        group_stmt = (
            select(
                Question.quiz_group_id,
                Question.round_number,
                func.min(Question.created_at).label("created_at"),
                func.count(Question.id).label("q_num")
            )
            .join(ImportanceResult, Question.importance_id == ImportanceResult.id)
            .join(DocumentChunk, ImportanceResult.chunk_id == DocumentChunk.id)
            .where(DocumentChunk.document_id.in_(sibling_ids))
            .where(Question.quiz_group_id.isnot(None))
            .group_by(Question.quiz_group_id, Question.round_number)
            .order_by(Question.round_number.desc())
        )
        group_rows = (await db.execute(group_stmt)).all()

        attempts = []
        for group in group_rows:
            qr_stmt = (
                select(QuizResult)
                .where(QuizResult.document_id.in_(sibling_ids))
                .where(QuizResult.quiz_group_id == group.quiz_group_id)
                .order_by(QuizResult.created_at.desc())
                .limit(1)
            )
            qr = (await db.execute(qr_stmt)).scalar_one_or_none()

            attempts.append(AttemptItem(
                quiz_result_id=qr.id if qr else None,
                quiz_group_id=group.quiz_group_id,
                round=group.round_number if group.round_number else 1,
                created_at=group.created_at.isoformat(),
                q_num=group.q_num,
                score=qr.score_percent if qr and qr.correct_count > 0 else None,
                attempt_phase=qr.attempt_phase if qr else None,
            ))

        documents.append(DocumentItem(
            document_id=doc.id,
            group_id=doc.group_id,
            title=doc.title,
            upload_date=doc.created_at.isoformat() if doc.created_at else "",
            total_count=len(group_rows),
            attempts=attempts,
        ))

    return QuestionListResponse(documents=documents)

# ────────────────────────────────────────
# 6. 회차 삭제
# ────────────────────────────────────────
async def delete_quiz_results_service(
    request: DeleteQuizResultRequest,
    db: AsyncSession,
) -> DeleteQuizResultResponse:
    from db.models import UserAnswer, QuizResult
    from uuid import UUID as UUIDType

    deleted_count = 0

    # ── quiz_result_ids만으로 삭제 (채점 기록만, 문제 유지) ──
    for qr_id in request.quiz_result_ids:
        qr_stmt = select(QuizResult).where(QuizResult.id == qr_id)
        qr = (await db.execute(qr_stmt)).scalar_one_or_none()
        if qr:
            if qr.user_id != request.user_id:
                raise PermissionError("삭제 권한이 없습니다.")
            await db.delete(qr)
            deleted_count += 1

    # ── quiz_group_ids 기준 삭제 (문제 + 채점 기록 모두) ──
    for group_id_str in request.quiz_group_ids:
        group_id = UUIDType(group_id_str)

        qr_stmt = select(QuizResult).where(QuizResult.quiz_group_id == group_id)
        quiz_results = (await db.execute(qr_stmt)).scalars().all()

        for qr in quiz_results:
            if qr.user_id != request.user_id:
                raise PermissionError("삭제 권한이 없습니다.")
            await db.delete(qr)

        q_stmt = select(Question).where(Question.quiz_group_id == group_id)
        questions = (await db.execute(q_stmt)).scalars().all()
        for q in questions:
            await db.delete(q)

        deleted_count += 1

    await db.commit()

    return DeleteQuizResultResponse(
        deleted_count=deleted_count,
        message="성공적으로 삭제되었습니다."
    )

# ────────────────────────────────────────
# 7. quiz_group_id로 문제 조회
# ────────────────────────────────────────
async def get_questions_by_group_service(
    quiz_group_id: str,
    db: AsyncSession,
) -> QuestionsByGroupResponse:
    from uuid import UUID as UUIDType

    stmt = (
        select(Question, ImportanceResult, DocumentChunk)
        .join(ImportanceResult, Question.importance_id == ImportanceResult.id)
        .join(DocumentChunk, ImportanceResult.chunk_id == DocumentChunk.id)
        .where(Question.quiz_group_id == UUIDType(quiz_group_id))
        .order_by(Question.id.asc())
    )
    rows = (await db.execute(stmt)).all()

    if not rows:
        raise ValueError("해당 문제 그룹을 찾을 수 없습니다.")

    questions = []
    for idx, (question, importance, chunk) in enumerate(rows):
        source_type = get_source_type(chunk.meta_data or [])
        priority = int(question.difficulty) if question.difficulty else 3
        options = question.options if question.options else None

        questions.append(QuestionItem(
            chunk_id=chunk.id,
            question_id=question.id,
            question_type=question.question_type,
            question_text=question.question_text,
            options=options,
            answer=question.answer,
            explanation=question.explanation,
            question_number=idx + 1,
            priority=priority,
            source_type=source_type,
            page_number=chunk.page_number,
        ))

    return QuestionsByGroupResponse(
        quiz_group_id=UUIDType(quiz_group_id),
        questions=questions,
    )

# ────────────────────────────────────────
# 8. 오답 조회
# ────────────────────────────────────────
async def get_wrong_answers_service(
    quiz_result_id: int,
    db: AsyncSession,
):
    from db.models import UserAnswer
    from agents.question_agent.schemas import WrongAnswerItem, WrongAnswersResponse

    stmt = (
        select(UserAnswer, Question, ImportanceResult, DocumentChunk)
        .join(Question, UserAnswer.question_id == Question.id)
        .join(ImportanceResult, Question.importance_id == ImportanceResult.id)
        .join(DocumentChunk, ImportanceResult.chunk_id == DocumentChunk.id)
        .where(UserAnswer.quiz_result_id == quiz_result_id)
        .where(UserAnswer.is_correct == False)
    )
    rows = (await db.execute(stmt)).all()

    wrong_answers = []
    for user_answer, question, importance, chunk in rows:
        priority = int(question.difficulty) if question.difficulty else 3

        order_stmt = (
            select(func.count(Question.id))
            .where(Question.quiz_group_id == question.quiz_group_id)
            .where(Question.id <= question.id)
        )
        question_number = (await db.execute(order_stmt)).scalar() or 0

        wrong_answers.append(WrongAnswerItem(
            question_id=question.id,
            question_number=question_number,
            question_type=question.question_type,
            question_text=question.question_text,
            options=question.options,
            answer=question.answer,
            explanation=question.explanation,
            submitted_answer=user_answer.user_answer or "",
            page_number=chunk.page_number,
            priority=priority,
        ))

    return WrongAnswersResponse(
        quiz_result_id=quiz_result_id,
        wrong_answers=wrong_answers,
    )

# ────────────────────────────────────────
# 9. 채점 결과 상세 조회
# ────────────────────────────────────────
async def get_quiz_result_detail_service(
    quiz_result_id: int,
    db: AsyncSession,
):
    from db.models import UserAnswer, QuizResult
    from agents.question_agent.schemas import QuizResultDetailResponse

    qr_stmt = select(QuizResult).where(QuizResult.id == quiz_result_id)
    qr = (await db.execute(qr_stmt)).scalar_one_or_none()
    if not qr:
        raise ValueError("채점 결과를 찾을 수 없습니다.")

    ua_stmt = (
        select(UserAnswer, Question)
        .join(Question, UserAnswer.question_id == Question.id)
        .where(UserAnswer.quiz_result_id == quiz_result_id)
    )
    rows = (await db.execute(ua_stmt)).all()

    results = []
    for user_answer, question in rows:
        results.append(AnswerResult(
            question_id=question.id,
            submitted_answer=user_answer.user_answer or "",
            correct_answer=question.answer,
            is_correct=user_answer.is_correct,
            explanation=question.explanation,
        ))

    return QuizResultDetailResponse(
        total=qr.total_questions,
        correct=qr.correct_count,
        wrong=qr.total_questions - qr.correct_count,
        results=results,
    )

# ────────────────────────────────────────
# 10. 문서 폴더명 변경
# ────────────────────────────────────────
async def update_document_title_service(request: dict, db: AsyncSession):
    from uuid import UUID as UUIDType

    group_id = request.get("group_id")
    new_title = request.get("title")
    user_id = request.get("user_id")

    stmt = select(Document).where(
        Document.group_id == UUIDType(group_id),
        Document.user_id == user_id
    )
    docs = (await db.execute(stmt)).scalars().all()

    if not docs:
        raise ValueError("문서를 찾을 수 없습니다.")

    for doc in docs:
        doc.title = new_title

    await db.commit()
    return {"message": "제목이 수정되었습니다.", "title": new_title}