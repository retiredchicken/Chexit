import io
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from app.chexit_inference import predict_chexit_from_pil_rgb


def _cors_origins() -> list[str]:
    raw = os.environ.get(
        "CHEXIT_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,https://your-vercel-site.vercel.app",
    )
    return [o.strip() for o in raw.split(",") if o.strip()]


app = FastAPI(title="Chexit API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictResponse(BaseModel):
    diagnosis: str
    risk_score: float = Field(..., description="Estimated TB probability (0–100)")
    confidence_label: str
    heatmap: str = Field(..., description="PNG overlay (CAM on CXR) as base64")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file.",
        )

    file_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image.")

    try:
        out = predict_chexit_from_pil_rgb(image)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e!s}") from e

    return PredictResponse(
        diagnosis=str(out["diagnosis"]),
        risk_score=float(out["risk_score"]),
        confidence_label=str(out["confidence_label"]),
        heatmap=str(out["heatmap"]),
    )
