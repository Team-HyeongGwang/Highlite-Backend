# Few-shot 예시를 캐시에 저장

import base64
import fitz
import datetime
import google.generativeai as genai
from google.generativeai import caching
from db.supabase_client import get_supabase_client
from service import SYSTEM_PROMPT

# ── 전역 상태 ──────────────────────────────────────────
_cache = None

# ── Few-shot 파일 목록 ──────────────────────────────────────────
FEW_SHOT_FILES = [
    "handwritten_clean",
    "handwritten_formula",
    "slide-formula",
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