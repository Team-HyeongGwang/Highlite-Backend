# 프롬프트, Few-shot 예시를 캐시에 저장

import base64
import fitz
import datetime
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
- [필기/형광펜]: 슬라이드에 손글씨(handwriting)나 형광펜(highlight)이 칠해져 있다면, 그 텍스트나 의미를 `content`에 자연스럽게 포함시킵니다. 그리고 반드시 해당 색상을 별도 필드에 명시하세요.
- [밑줄]: 손으로 그은 밑줄이 있으면 해당 텍스트를 [밑줄: 내용] 형태로 감싸고, handwriting_color에 색상을 명시하세요.
- [동그라미/네모]: 특정 단어나 구문에 동그라미나 네모박스가 쳐져 있으면 [동그라미: 내용] 또는 [네모: 내용] 형태로 감싸고, handwriting_color에 색상을 명시하세요.
- [손필기]: 슬라이드 여백에 손으로 적은 메모가 있으면 [필기: 내용] 형태로 content에 포함시키고, handwriting_color에 색상을 명시하세요.

[출력 형식]
반드시 아래 JSON 배열(List) 형식으로만 응답하세요. 다른 텍스트는 금지입니다.
[
  {
      "paragraph_index": 1,
      "content": "추출된 텍스트 내용 또는 이미지/도표에 대한 설명",
      "handwriting_color": null,
      "highlight_color": null
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
]

# ── 유틸 ──────────────────────────────────────────
def pdf_page_to_b64(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pix = doc[0].get_pixmap(dpi=150)
    return base64.b64encode(pix.tobytes("jpeg")).decode("utf-8")


# ── 서버 시작 시 한 번만 실행 ──────────────────────────────────────────
async def run_primer(input_data: dict) -> dict:
    global _cache

    supabase = get_supabase_client()
    contents = []

    for name in FEW_SHOT_FILES:
        pdf_bytes = supabase.storage.from_("few-shot-examples").download(f"{name}.pdf")
        b64 = pdf_page_to_b64(pdf_bytes)
        answer_bytes = supabase.storage.from_("few-shot-examples").download(f"{name}.json")
        answer_text = answer_bytes.decode("utf-8")

        contents.append({"role": "user", "parts": [
            {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
            {"text": "페이지 1 텍스트를 추출해 주세요."}
        ]})
        contents.append({"role": "model", "parts": [{"text": answer_text}]})

    _cache = caching.CachedContent.create(
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