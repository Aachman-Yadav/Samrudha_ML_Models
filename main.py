from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import numpy as np
import joblib
from typing import cast
from keras.models import load_model, Model

app = FastAPI()

# === CORS Configuration ===
origins = [
    "http://localhost:3000",  # Dev
    "https://your-frontend.vercel.app"  # Replace with your deployed frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Model Loading ===
models_dir = Path(__file__).parent / "models"

keras_model = cast(Model, load_model(models_dir / "model.h5"))

crop_model = joblib.load(models_dir / "crop_recommendation.joblib")
fertilizer_model = joblib.load(models_dir / "fertilizer_recommendation.joblib")
yield_model = joblib.load(models_dir / "yield_prediction.joblib")

# 👇 Explicit typing fixes Pylance
# keras_model: Model = load_model(models_dir / "model.h5")

# === Health Check ===
@app.get("/health")
async def health_check():
    return {"status": "ok"}


# === Crop Recommendation ===
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


@app.post("/predict/crop")
async def predict_crop(input: CropInput):
    X = np.array([[input.N, input.P, input.K, input.temperature, input.humidity, input.ph, input.rainfall]])
    prediction = crop_model.predict(X)[0]
    return {"crop": prediction}


# === Fertilizer Recommendation ===
class FertilizerInput(BaseModel):
    temperature: float
    humidity: float
    moisture: float
    soil_type: int
    crop_type: int


@app.post("/predict/fertilizer")
async def predict_fertilizer(input: FertilizerInput):
    X = np.array([[input.temperature, input.humidity, input.moisture, input.soil_type, input.crop_type]])
    prediction = fertilizer_model.predict(X)[0]
    return {"fertilizer": prediction}


# === Yield Prediction ===
class YieldInput(BaseModel):
    rainfall: float
    temperature: float
    humidity: float
    area: float


@app.post("/predict/yield")
async def predict_yield(input: YieldInput):
    X = np.array([[input.rainfall, input.temperature, input.humidity, input.area]])
    prediction = yield_model.predict(X)[0]
    return {"predicted_yield": prediction}


# === Keras Model Prediction ===
class KerasInput(BaseModel):
    feature1: float
    feature2: float
    # Add more fields if needed by your .h5 model


@app.post("/predict/keras")
async def predict_keras(input: KerasInput):
    X = np.array([[input.feature1, input.feature2]])  # match model input shape
    prediction = keras_model.predict(X).tolist()
    return {"keras_output": prediction}
