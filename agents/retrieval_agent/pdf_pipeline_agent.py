# PDF 읽기
# 페이지별 텍스트 + VisualCue(형광펜/필기) 추출
# List[ChunkExtraction] 반환

from schemas import ChunkExtraction

# pdf_pipeline_agent.py가 반환하는 것
List[ChunkExtraction]

# ChunkExtraction 예시
ChunkExtraction(
    page_number=1,
    original_text="미토콘드리아는 세포의 발전소다...",
    meta_data=[
        VisualCue(type="highlight", color="yellow", target_text="미토콘드리아"),
        VisualCue(type="pen", color="red", target_text="발전소"),
        VisualCue(type="memo", color="blue", target_text="시험에 나옴!!")
    ]
)