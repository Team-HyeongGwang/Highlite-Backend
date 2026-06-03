import base64
import fitz
import datetime
import asyncio
import google.generativeai as genai
from google.generativeai import caching
from db.supabase_client import get_supabase_client

SYSTEM_PROMPT = """
당신은 강의 자료(PDF 슬라이드)를 분석하여 RAG 시스템에 활용될 청크(Chunk) 단위 데이터로 변환하는 AI입니다.

[기본 원칙]
1. 텍스트 추출 전, 슬라이드의 전체적인 맥락과 흐름을 먼저 파악하고 이를 바탕으로 청킹(Chunking)하세요.
2. 당신의 분석 과정, 생각, 부연 설명은 절대 출력하지 마세요. 오직 요청된 JSON 배열 형태만 반환해야 합니다.
3. 슬라이드 상/하단의 페이지 번호, 강의명 등 본문과 무관한 텍스트는 추출에서 제외하세요.

[데이터 처리 가이드]
1. 청킹(Chunking) 기준
- '의미상 이어지는 단일 개념'이나 '문단' 단위로 묶어주세요. 단어나 짧은 줄 단위로 파편화하지 마세요.
- 위에서부터 아래로 자연스러운 흐름에 따라 `paragraph_index`를 1부터 순차적으로 부여하세요.

2. 요소별 추출 규칙 (content 작성)
- [일반 텍스트]: 슬라이드에 적힌 실제 텍스트를 문맥에 맞게 묶어 `content`에 작성합니다.
- [이미지/도표]: 그림이나 도표 안에 포함된 글자들을 따로 떼어내지 마세요. 도표의 의미와 포함된 텍스트를 종합하여 객관적인 묘사로 `content`에 작성합니다.
- [필기/형광펜 기본]: 슬라이드에 필기가 있다면 해당 색상을 `handwriting_color`에, 형광펜이 있다면 `highlight_color`에 반드시 명시하세요.
- [밑줄]: 손으로 그은 밑줄이 있다면 해당 텍스트를 [밑줄: 내용] 형태로 감싸세요.
- [동그라미/네모]: 특정 단어나 구문에 도형 표시가 있다면 [강조도형: 내용] 형태로 감싸세요.
- [최상위 강조 표시]: 텍스트 주변에 별표(★) 기호나 '중요', '시험', '핵심', '출제' 등 중요도를 극도로 강조하는 필기가 있다면, 학생이 적은 그 단어/기호를 그대로 살려 [별표: 내용], [시험: 내용], [핵심: 내용] 형태로 감싸세요.
  ⛔️ 주의: "이 이론의 핵심은~" 또는 "시험에 자주 나오는~" 처럼 문장(본문 또는 일반 필기) 안에 자연스럽게 포함된 단어는 절대 태그로 변환하지 마세요. 오직 특정 텍스트를 강조하기 위해 시각적으로 덧붙여진 마커(Marker) 용도일 때만 태그를 적용해야 합니다.
- [단순 메모]: 본문에 없는 내용을 여백에 추가로 필기했다면 문맥에 맞는 위치에 [필기: 내용] 형태로 삽입하세요.

[출력 형식 및 예시]
반드시 아래 JSON 배열(List) 형태의 '예시'를 참고하여, 실제 추출한 슬라이드 내용으로 값을 채워 응답하세요. JSON 외의 다른 설명 텍스트는 절대 금지입니다.

[
  {
      "paragraph_index": 1,
      "content": "이 슬라이드는 RAG 아키텍처의 기본 구조를 설명하는 다이어그램입니다.",
      "handwriting_color": null,
      "highlight_color": null
  },
  {
      "paragraph_index": 2,
      "content": "[별표: RAG 시스템]의 가장 큰 장점은 [밑줄: 환각 현상을 줄일 수 있다]는 것입니다. [필기: 시험 출제 확률 높음]",
      "handwriting_color": "red",
      "highlight_color": "yellow"
  }
]
"""

# ── 전역 상태 ──────────────────────────────────────────
_cache = None

# ── Few-shot 파일 목록 ──────────────────────────────────────────
FEW_SHOT_FILES = [
    "handwritten_clean",
    "handwritten_formula",
    "slide_formula",
    "slide_dense",
    "slide_symbol"
]

# ── 유틸 ──────────────────────────────────────────
def pdf_page_to_b64(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pix = doc[0].get_pixmap(dpi=150)
    return base64.b64encode(pix.tobytes("jpeg")).decode("utf-8")

# ── 개별 파일 1세트(PDF + JSON)를 비동기로 다운로드하고 변환하는 내부 함수 ──
async def _download_and_process_file(supabase, name: str) -> list:
    # 1) 스토리지에서 PDF와 JSON을 동시에(병렬) 다운로드 시도
    # 루프 바깥의 메인 스레드를 막지 않고 비동기로 처리
    pdf_task = asyncio.to_thread(supabase.storage.from_("few-shot-examples").download, f"{name}.pdf")
    json_task = asyncio.to_thread(supabase.storage.from_("few-shot-examples").download, f"{name}.json")
    
    pdf_bytes, answer_bytes = await asyncio.gather(pdf_task, json_task)
    
    # 2) CPU 연산(이미지 변환)은 루프를 안 막도록 별도 스레드에서 처리하면 좋으나, 1장이므로 바로 처리
    b64 = pdf_page_to_b64(pdf_bytes)
    answer_text = answer_bytes.decode("utf-8")
    
    # 제미나이 컨텐츠 포맷 구조 반환
    return [
        {"role": "user", "parts": [
            {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
            {"text": "페이지 1 텍스트를 추출해 주세요."}
        ]},
        {"role": "model", "parts": [{"text": answer_text}]}
    ]


# ── 서버 시작 시 한 번만 실행 ──────────────────────────────────────────
async def run_primer(input_data: dict) -> dict:
    global _cache

    supabase = get_supabase_client()
    
    print(f"[Primer] Few-shot 파일 {len(FEW_SHOT_FILES)}개 병렬 다운로드 시작... 🚀")
    
    # 1) 4개의 파일 쌍을 가져오는 태스크를 동시에 생성
    tasks = [_download_and_process_file(supabase, name) for name in FEW_SHOT_FILES]
    
    # 2) 4개 세트가 동시에 네트워크 통신을 수행하도록 병렬 실행
    results = await asyncio.gather(*tasks)
    
    # 3) 병렬로 받아온 결과들을 하나의 contents 리스트로 병합
    contents = []
    for pair in results:
        contents.extend(pair)
        
    print(f"[Primer] Few-shot 컨텐츠 조립 완료. 제미나이 컨텍스트 캐시 생성 중... 🧠")
    
    # 4) 제미나이 컨텍스트 캐시 생성
    # Google 서버와 통신하는 동기(Sync) 함수를 asyncio.to_thread로 감싸서 비동기로 처리
    _cache = await asyncio.to_thread(
        caching.CachedContent.create,
        model="models/gemini-2.5-pro",
        system_instruction=SYSTEM_PROMPT,
        contents=contents,
        ttl=datetime.timedelta(hours=6),
    )
    
    print(f"[Primer] 캐시 등록 완료 — cache name: {_cache.name}")
    return input_data


def get_cache():
    if not _cache:
        raise RuntimeError("Cache가 초기화되지 않았습니다. run_primer()를 먼저 실행하세요.")
    return _cache