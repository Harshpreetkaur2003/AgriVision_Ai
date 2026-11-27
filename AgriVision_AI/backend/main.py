
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.routes.prediction import (
    router as prediction_router,
    MODEL_PATH,
    CLASS_PATH
)

BASE_DIR = Path(__file__).resolve().parent.parent

FRONTEND_DIR = BASE_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"

app = FastAPI(
    title="AgriVision AI",
    description="AI-powered Crop Disease Detection System",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)

app.include_router(prediction_router)


@app.get("/", include_in_schema=False)
async def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/analysis", include_in_schema=False)
async def analysis():
    return FileResponse(FRONTEND_DIR / "analysis.html")


@app.get("/health")
async def health():
    return {
        "status": "running",
        "project": "AgriVision AI",
        "phase": 7,
        "model_file_exists": MODEL_PATH.exists(),
        "class_mapping_exists": CLASS_PATH.exists()
    }
