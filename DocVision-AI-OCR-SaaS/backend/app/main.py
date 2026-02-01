from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .core.config import settings
from .api.routes import router
import os

app = FastAPI(title="DocVision AI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(settings.output_dir, exist_ok=True)
os.makedirs(settings.tmp_dir, exist_ok=True)
os.makedirs(settings.upload_tmp_dir, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=settings.output_dir), name="outputs")

app.include_router(router, prefix="/api")
