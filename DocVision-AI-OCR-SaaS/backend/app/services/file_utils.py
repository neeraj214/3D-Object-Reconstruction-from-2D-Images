import os
import uuid
from fastapi import UploadFile


async def save_upload_file(file: UploadFile, target_dir: str) -> str:
    os.makedirs(target_dir, exist_ok=True)
    ext = os.path.splitext(file.filename)[1].lower()
    name = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(target_dir, name)
    with open(path, "wb") as f:
        f.write(await file.read())
    return path
