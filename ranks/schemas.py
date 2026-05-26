from pydantic import BaseModel, Field
from typing import Dict, Optional

class RankingRequest(BaseModel):
    highlighter_ranking: Optional[Dict[str, int]] = Field(None, description="형광펜 중요도 순위")
    pen_ranking: Optional[Dict[str, int]] = Field(None, description="펜 중요도 순위")