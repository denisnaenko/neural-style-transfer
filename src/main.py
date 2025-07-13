from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def home():
    return {"home": "page"}


@app.get("/healthz")
def read_api_healt():
    return {"status": "ok"}
