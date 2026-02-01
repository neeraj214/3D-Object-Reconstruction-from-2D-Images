import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    output_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "output"))
    tmp_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tmp"))
    upload_tmp_dir: str = os.path.join(tmp_dir, "upload_tmp")
    default_lang: str = "en"


settings = Settings()
