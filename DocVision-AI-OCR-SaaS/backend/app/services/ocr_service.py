import time
from typing import Any, Dict, List
import easyocr
from .preprocessing_service import preprocess_image


def run_ocr(path: str, original_filename: str) -> Dict[str, Any]:
    start = time.perf_counter()
    img = preprocess_image(path)
    reader = easyocr.Reader(["en"], gpu=False)
    results = reader.readtext(img)
    blocks: List[Dict[str, Any]] = []
    for r in results:
        bbox = r[0]
        text = r[1]
        conf = float(r[2]) if len(r) > 2 else 0.0
        blocks.append({"text": text, "confidence": conf, "bbox": [[float(p[0]), float(p[1])] for p in bbox]})
    full_text = "\n".join([b["text"] for b in blocks])
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return {
        "status": "success",
        "text": full_text,
        "blocks": blocks,
        "metadata": {"filename": original_filename, "processing_time_ms": elapsed_ms},
    }
