"""FastAPI service for heart disease prediction."""

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator


# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "artifacts" / "best_model.joblib"

app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "Predicts the model's heart disease classification "
        "from patient feature values."
    ),
    version="1.0.0",
)

# Expose Prometheus request metrics at /metrics.
Instrumentator().instrument(app).expose(
    app,
    endpoint="/metrics",
    include_in_schema=False,
)

# Load the trained pipeline once when the API module starts.
model = None
model_load_error = None

try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

except Exception as exc:
    model_load_error = str(exc)


class PatientFeatures(BaseModel):
    """Input features expected by the trained model."""

    age: int = Field(ge=1, le=120)
    sex: Literal[0, 1]
    cp: int = Field(ge=1, le=4)
    trestbps: float = Field(gt=0)
    chol: float = Field(ge=0)
    fbs: Literal[0, 1]
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(gt=0)
    exang: Literal[0, 1]
    oldpeak: float
    slope: int = Field(ge=1, le=3)
    ca: float = Field(ge=0, le=3)
    thal: int = Field(ge=3, le=7)

@app.get("/")
def home():
    return {
        "message": "Heart Disease Prediction API",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
    }

@app.get("/health")
def health():
    """Check whether the API process is running."""

    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Check whether the model is available for predictions."""

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not loaded: {model_load_error}",
        )

    return {"status": "ready", "model": "best_model.joblib"}


@app.post("/predict")
def predict(patient: PatientFeatures):
    """Return the model prediction and positive-class probability."""

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is unavailable. Check the model artifact.",
        )

    try:
        input_data = pd.DataFrame(
            [patient.model_dump()]
        )

        prediction = int(model.predict(input_data)[0])

        probabilities = model.predict_proba(input_data)[0]
        classes = list(model.classes_)
        positive_index = classes.index(1)
        probability = float(probabilities[positive_index])

        return {
            "prediction": prediction,
            "label": (
                "Heart disease detected by model"
                if prediction == 1
                else "No heart disease detected by model"
            ),
            "positive_class_probability": round(probability, 4),
            "disclaimer": (
                "This is a machine-learning prediction, "
                "not a medical diagnosis."
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc