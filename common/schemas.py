from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class VisualCue(BaseModel):
    """시각적 강조 정보 (형광펜, 펜 등)"""
    type: str = Field(..., description="'highlight', 'pen', 'memo'")
    color: str = Field(..., description="색상 (yellow, red 등)")
    target_text: str = Field(..., description="강조된 텍스트")