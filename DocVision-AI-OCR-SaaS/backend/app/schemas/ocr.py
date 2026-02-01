from pydantic import BaseModel
from typing import List


class OCRBlock(BaseModel):
    text: str
    confidence: float
    bbox: List[List[float]]


class OCRMetadata(BaseModel):
    filename: str
    processing_time_ms: int


class OCRV1Response(BaseModel):
    status: str
    text: str
    blocks: List[OCRBlock]
    metadata: OCRMetadata
