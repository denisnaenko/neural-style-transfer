import shutil
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from style_transfer.service import process_style_transfer

app = FastAPI()

RESULT_DIR = Path(__file__).parent / "result"
RESULT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

templates = Jinja2Templates(directory="templates")

STYLE_PATH = Path(__file__).parent / "styles" / "style.jpg"
CONTENT_PATH = UPLOAD_DIR / "content.jpg"
RESULT_PATH = RESULT_DIR / "result.jpg"
PLOT_PATH = RESULT_DIR / "loss_plot.png"


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
    if not file.filename:
        return JSONResponse(status_code=400, content={"detail": "Файл без имени."})

    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    shutil.copy(file_path, UPLOAD_DIR / "content.jpg")
    return JSONResponse(
        status_code=200,
        content={
            "filename": file.filename,
            "url": f"/uploads/{file.filename}",
            "detail": "Файл успешно загружен.",
        },
    )


@app.get("/uploads/{filename}")
def get_uploaded_file(filename: str):
    file_path = UPLOAD_DIR / filename
    return FileResponse(file_path)


@app.post("/transfer/")
async def transfer_style():
    try:
        _ = process_style_transfer(
            str(CONTENT_PATH),
            str(Path(__file__).parent / "styles"),
            str(RESULT_PATH),
            str(PLOT_PATH),
            resize_size=(512, 512),
        )
        return {"result": "/result/", "plot": "/plot/"}
    except Exception as e:
        import traceback

        print("[ERROR in /transfer/]:", e)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": str(e)})


@app.get("/result/")
def get_result():
    return FileResponse(RESULT_PATH, media_type="image/jpeg")


@app.get("/plot/")
def get_plot():
    return FileResponse(PLOT_PATH, media_type="image/png")


@app.get("/healthz")
def read_api_healt():
    return {"status": "ok"}
