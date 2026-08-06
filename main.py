"""
Main FastAPI server for the DeepFake & Misinformation Fact-Checker.

Run locally:
    uvicorn main:app --reload --port 8000

Endpoints:
    GET  /health              -> simple check that the server is alive
    POST /check-text          -> { "text": "..." }  -> credibility result
    POST /check-image         -> multipart file upload -> credibility result
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from text_checker import evaluate_text
from image_checker import evaluate_image

app = FastAPI(title="Fact-Checker API", version="0.1.0")

# The Chrome extension calls this API from a content script running on
# arbitrary websites, so we allow all origins. Tighten this if you ever
# ship it publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/check-text")
def check_text(payload: TextRequest):
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="text field is empty")
    return evaluate_text(payload.text)


@app.post("/check-image")
async def check_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="uploaded file is not an image")
    image_bytes = await file.read()
    return evaluate_image(image_bytes)

