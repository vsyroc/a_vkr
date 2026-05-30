from contextlib import asynccontextmanager
import uuid
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.ai import get_model, run_inference, get_lama, run_inpainting, run_segmentation
from app.config import BASE_DIR, UPLOADS_DIR, PROCESSED_DIR


@asynccontextmanager
async def lifespan(app):
    app.state.model = get_model()
    get_lama()
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


@app.post("/api/process") @ app.post("/api/process")
async def process_image(request: Request, file: UploadFile = File(...)):
    ext = Path(file.filename).suffix or ".jpg"
    file_id = uuid.uuid4().hex

    original_path = UPLOADS_DIR / f"{file_id}{ext}"
    mask_path = PROCESSED_DIR / f"{file_id}_mask.jpg"

    with original_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        run_segmentation(request.app.state.model, original_path, mask_path)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    return JSONResponse(
        {
            "file_id": file_id,
            "original_url": f"/uploads/{original_path.name}",
            "mask_url": f"/processed/{mask_path.name}",
        }
    )


@app.post("/api/inpaint")
async def inpaint_image(request: Request, body: dict):
    file_id = body.get("file_id")
    if not file_id:
        return JSONResponse({"error": "file_id required"}, status_code=400)

    originals = list(UPLOADS_DIR.glob(f"{file_id}.*"))
    masks = list(PROCESSED_DIR.glob(f"{file_id}_mask.*"))

    if not originals or not masks:
        return JSONResponse({"error": "Files not found"}, status_code=404)

    inpainted_path = PROCESSED_DIR / f"{file_id}_inpainted.jpg"

    try:
        run_inpainting(originals[0], masks[0], inpainted_path)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    return JSONResponse(
        {
            "inpainted_url": f"/processed/{inpainted_path.name}",
        }
    )
