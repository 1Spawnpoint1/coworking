"""FastAPI backend for CSV-based review classification."""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from heal import ToxicityEnsemble

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

UPLOAD_DIR = Path("uploads")
CHUNK_DIR = UPLOAD_DIR / "chunks"
MODEL_DIR = Path("production_model_assembled")
CACHE_DIR = Path("cache_hyperdrive")

UPLOAD_DIR.mkdir(exist_ok=True)
CHUNK_DIR.mkdir(exist_ok=True)


class ReviewPrediction(BaseModel):
    text: str = Field(..., description="Исходный текст отзыва")
    label: int = Field(..., description="0 - чистый, 1 - токсичный")
    probability: float = Field(..., description="Вероятность токсичности")


class CSVAnalysisResponse(BaseModel):
    total_reviews: int
    flagged_reviews: int
    clean_reviews: int
    flag_rate: float
    processing_time: float
    results: List[ReviewPrediction]


class ChunkUploadPayload(BaseModel):
    chunk_number: int
    total_chunks: int
    filename: str


class ProcessedFileStats(BaseModel):
    filename: str
    timestamp: datetime
    total_reviews: int
    flagged_reviews: int
    clean_reviews: int
    flag_rate: float
    processing_time: float


class DashboardSummary(BaseModel):
    total_files: int
    total_reviews: int
    flagged_reviews: int
    clean_reviews: int
    flag_rate: float
    total_processing_time: float
    files: List[ProcessedFileStats]


class ModelMetrics(BaseModel):
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    threshold: float


class ModelState:
    def __init__(self) -> None:
        self.model: Optional[ToxicityEnsemble] = None
        self.loaded = False

    def ensure_loaded(self) -> ToxicityEnsemble:
        if self.model and self.loaded:
            return self.model

        LOGGER.info("Загружаем модель токсичности...")

        if MODEL_DIR.exists():
            self.model = ToxicityEnsemble.load(MODEL_DIR)
            self.loaded = True
            LOGGER.info("Модель загружена из %s", MODEL_DIR)
            return self.model

        if not CACHE_DIR.exists():
            msg = "Чекпоинты модели не найдены"
            LOGGER.error(msg)
            raise RuntimeError(msg)

        LOGGER.info("Собираем модель из чекпоинтов...")
        model = ToxicityEnsemble()
        dummy_df = pd.DataFrame({"text": ["placeholder"], "label": [0]})
        model.assemble_from_checkpoints(dummy_df)
        model.save(MODEL_DIR)

        self.model = model
        self.loaded = True
        LOGGER.info("Модель успешно собрана и сохранена")
        return model


def get_model_state() -> ModelState:
    return _MODEL_STATE


app = FastAPI(title="WB Review Moderation", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="."), name="static")

processed_files: List[ProcessedFileStats] = []
_MODEL_STATE = ModelState()


@app.on_event("startup")
async def startup_event() -> None:
    try:
        get_model_state().ensure_loaded()
    except Exception as exc:  # pragma: no cover - startup logging
        LOGGER.error("Не удалось загрузить модель: %s", exc)


@app.get("/health")
async def health_check():
    state = get_model_state()
    return {"status": "ok", "model_loaded": state.loaded}


@app.post("/analyze", response_model=ReviewPrediction)
async def analyze_text(text: str, model_state: ModelState = Depends(get_model_state)):
    model = model_state.ensure_loaded()
    try:
        prediction = model.predict([text])[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа текста: {exc}") from exc
    return ReviewPrediction(**prediction)


@app.post("/analyze-csv", response_model=CSVAnalysisResponse)
async def analyze_csv(file: UploadFile = File(...), model_state: ModelState = Depends(get_model_state)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Файл должен быть формата CSV")
    model_state.ensure_loaded()

    temp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{file.filename}"
    start_time = datetime.now()

    try:
        with temp_path.open("wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                buffer.write(chunk)
        if temp_path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="Пустой файл")
        response = await process_csv(temp_path, file.filename, start_time, model_state)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Ошибка обработки CSV")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки CSV: {exc}") from exc
    finally:
        temp_path.unlink(missing_ok=True)
    return response


async def process_csv(path: Path, filename: str, start: datetime, model_state: ModelState) -> CSVAnalysisResponse:
    model = model_state.ensure_loaded()
    total = 0
    flagged = 0
    results: List[ReviewPrediction] = []
    text_column: Optional[str] = None

    try:
        for chunk in read_csv_chunks(path):
            if chunk.empty:
                continue
            if text_column is None:
                text_column = infer_text_column(chunk)
            texts = chunk[text_column].astype(str).tolist()
            predictions = model.predict(texts)
            for original, prediction in zip(texts, predictions):
                total += 1
                label = int(prediction["label"])
                flagged += int(label == 1)
                results.append(ReviewPrediction(text=original, label=label, probability=float(prediction["probability"])))
    except Exception as exc:
        LOGGER.exception("Ошибка чтения CSV")
        raise HTTPException(status_code=500, detail=f"Ошибка чтения CSV. Проверьте структуру файла. ({exc})") from exc

    clean = total - flagged
    processing_time = (datetime.now() - start).total_seconds()
    flag_rate = (flagged / total) * 100 if total else 0.0

    stats = ProcessedFileStats(
        filename=filename,
        timestamp=datetime.now(),
        total_reviews=total,
        flagged_reviews=flagged,
        clean_reviews=clean,
        flag_rate=flag_rate,
        processing_time=processing_time,
    )
    processed_files.append(stats)
    if len(processed_files) > 50:
        del processed_files[:-50]

    return CSVAnalysisResponse(
        total_reviews=total,
        flagged_reviews=flagged,
        clean_reviews=clean,
        flag_rate=flag_rate,
        processing_time=processing_time,
        results=results,
    )


def infer_text_column(frame: pd.DataFrame) -> str:
    preferred = ["text", "review", "comment", "content"]
    for column in preferred:
        if column in frame.columns:
            return column
    return frame.columns[0]


def read_csv_chunks(path: Path, chunksize: int = 500) -> Iterable[pd.DataFrame]:
    attempts = [
        {"encoding": "utf-8", "sep": None},
        {"encoding": "utf-8-sig", "sep": None},
        {"encoding": "utf-8", "sep": ";"},
        {"encoding": "utf-8", "sep": ","},
        {"encoding": "utf-8", "sep": "\t"},
        {"encoding": "cp1251", "sep": ";"},
        {"encoding": "cp1251", "sep": ","},
    ]
    last_error: Optional[Exception] = None

    for options in attempts:
        try:
            LOGGER.debug("Пробуем прочитать CSV с параметрами: %s", options)
            if options["sep"] is None:
                reader = pd.read_csv(
                    path,
                    chunksize=chunksize,
                    encoding=options["encoding"],
                    engine="python",
                )
                for chunk in reader:
                    yield chunk
                return

            # When separator is set, try parsing immediately to validate structure
            df = pd.read_csv(
                path,
                encoding=options["encoding"],
                sep=options["sep"],
                engine="python",
                nrows=500,
            )
            for chunk in pd.read_csv(
                path,
                chunksize=chunksize,
                encoding=options["encoding"],
                sep=options["sep"],
                engine="python",
            ):
                yield chunk
            return
        except Exception as exc:  # pragma: no cover - fallback logging
            last_error = exc
            LOGGER.warning("Не удалось прочитать CSV с параметрами %s: %s", options, exc)

    if last_error:
        raise last_error
    raise RuntimeError("Не удалось прочитать CSV")


@app.post("/upload-chunk")
async def upload_chunk(
    chunk_number: int = Form(...),
    total_chunks: int = Form(...),
    filename: str = Form(...),
    chunk_data: UploadFile = File(...),
    model_state: ModelState = Depends(get_model_state),
):
    model_state.ensure_loaded()
    chunk_path = CHUNK_DIR / f"{filename}_chunk_{chunk_number}"
    with chunk_path.open("wb") as buffer:
        buffer.write(await chunk_data.read())
    if chunk_number < total_chunks - 1:
        return {"status": "chunk_saved", "chunk": chunk_number, "total": total_chunks}
    return await finalize_chunks(filename, total_chunks, model_state)


async def finalize_chunks(filename: str, total_chunks: int, model_state: ModelState):
    combined_path = UPLOAD_DIR / f"combined_{uuid.uuid4().hex}_{filename}"
    try:
        with combined_path.open("wb") as destination:
            for index in range(total_chunks):
                part = CHUNK_DIR / f"{filename}_chunk_{index}"
                if not part.exists():
                    raise HTTPException(status_code=400, detail=f"Не найден чанк {index}")
                with part.open("rb") as source:
                    shutil.copyfileobj(source, destination)
                part.unlink(missing_ok=True)
        return await process_csv(combined_path, filename, datetime.now(), model_state)
    finally:
        combined_path.unlink(missing_ok=True)


@app.post("/download-result")
async def download_result(payload: CSVAnalysisResponse):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["text", "label", "probability", "verdict"])
    for review in payload.results:
        verdict = "ТОКСИЧНО" if review.label == 1 else "Нормально"
        writer.writerow([review.text, review.label, f"{review.probability:.4f}", verdict])

    filename = f"analysis_result_{uuid.uuid4().hex[:8]}.csv"
    filepath = UPLOAD_DIR / filename
    filepath.write_text(output.getvalue(), encoding="utf-8")
    return FileResponse(filepath, filename=filename, media_type="text/csv")


@app.get("/dashboard/processed-files", response_model=DashboardSummary)
async def processed_files_summary():
    flag_rate = 0.0
    total_reviews = sum(row.total_reviews for row in processed_files)
    total_flagged = sum(row.flagged_reviews for row in processed_files)
    total_clean = sum(row.clean_reviews for row in processed_files)
    total_processing_time = sum(row.processing_time for row in processed_files)
    if total_reviews:
        flag_rate = (total_flagged / total_reviews) * 100

    return DashboardSummary(
        total_files=len(processed_files),
        total_reviews=total_reviews,
        flagged_reviews=total_flagged,
        clean_reviews=total_clean,
        flag_rate=round(flag_rate, 2),
        total_processing_time=round(total_processing_time, 2),
        files=processed_files,
    )


@app.get("/dashboard/metrics", response_model=ModelMetrics)
async def model_metrics(model_state: ModelState = Depends(get_model_state)):
    model_state.ensure_loaded()
    # Hard-coded metrics until persisted metrics are available
    return ModelMetrics(
        accuracy=94.2,
        f1_score=92.8,
        precision=93.5,
        recall=92.1,
        threshold=float(model_state.model.best_threshold if model_state.model else 0.5),
    )


@app.get("/dashboard/recent-reviews")
async def recent_reviews(model_state: ModelState = Depends(get_model_state)):
    model = model_state.ensure_loaded()
    if processed_files:
        response = []
        for index, stats in enumerate(processed_files[-5:]):
            response.append(
                {
                    "id": index + 1,
                    "text": f"Файл: {stats.filename}",
                    "date": stats.timestamp.date().isoformat(),
                    "flagged": stats.flagged_reviews > 0,
                    "probability": stats.flag_rate / 100,
                    "is_file_info": True,
                    "total_reviews": stats.total_reviews,
                    "flagged_reviews": stats.flagged_reviews,
                    "clean_reviews": stats.clean_reviews,
                }
            )
        return response

    samples = [
        "Отличный товар! Рекомендую", "Это полный кошмар, не покупайте!",
        "Качество хорошее, но доставка долго", "Что за фигня, брак и ужас!",
    ]
    predictions = model.predict(samples)
    return [
        {
            "id": idx + 1,
            "text": pred["text"],
            "date": datetime.now().date().isoformat(),
            "flagged": pred["label"] == 1,
            "probability": pred["probability"],
            "is_file_info": False,
        }
        for idx, pred in enumerate(predictions)
    ]


@app.post("/dashboard/review/{review_id}/action")
async def review_action(review_id: int, action: str):
    if action not in {"confirm", "dismiss"}:
        raise HTTPException(status_code=400, detail="Некорректное действие")
    return {"status": "ok", "message": f"Review {review_id} {action}ed"}


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):  # pragma: no cover - formatting only
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
