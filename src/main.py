import shutil
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

templates = Jinja2Templates(directory="templates")


@app.get("/")
def home():
    return RedirectResponse("/upload")


@app.get("/upload")
def upload_form(request: Request):
    return templates.TemplateResponse("upload_form.html", {"request": request})


@app.post("/upload_image/")
async def upload_image(file: UploadFile = File(...)):
    # Проверка, что файл — изображение
    if not file.content_type or not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"detail": "Это не изображение."})

    # Проверка наличия имени файла
    if not file.filename:
        return JSONResponse(status_code=400, content={"detail": "Файл без имени."})

    # Cохраняем файл
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return JSONResponse(
        status_code=200,
        content={"filename": file.filename, "detail": "Файл успешно загружен."},
    )


@app.get("/healthz")
def read_api_healt():
    return {"status": "ok"}
