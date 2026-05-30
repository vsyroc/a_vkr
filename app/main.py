from contextlib import asynccontextmanager
import uuid
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.ai import get_model, run_inference
from app.config import BASE_DIR, UPLOADS_DIR, PROCESSED_DIR


@asynccontextmanager
async def lifespan(app):
    app.state.model = get_model()
    yield


app = FastAPI(title="ВКР — Обработка изображений", lifespan=lifespan)

UPLOADS_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")
app.mount("/processed", StaticFiles(directory=PROCESSED_DIR), name="processed")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/api/process")
async def process_image(request: Request, file: UploadFile = File(...)):
    ext = Path(file.filename).suffix or ".jpg"
    file_id = uuid.uuid4().hex

    original_path = UPLOADS_DIR / f"{file_id}{ext}"
    with original_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    processed_path = PROCESSED_DIR / f"{file_id}_processed.jpg"

    try:
        run_inference(request.app.state.model, original_path, processed_path)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    return JSONResponse(
        {
            "original_url": f"/uploads/{original_path.name}",
            "processed_url": f"/processed/{processed_path.name}",
        }
    )
