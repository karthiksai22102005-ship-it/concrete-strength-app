import os
import sys
import threading
import time
import webbrowser
import uvicorn
import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- 1. SETUP PATHS ---
if getattr(sys, 'frozen', False):
    base_dir = sys._MEIPASS
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))

static_dir = os.path.join(base_dir, "static")
templates_dir = os.path.join(base_dir, "templates")

app = FastAPI()

# --- 2. ALLOW CONNECTION (CORS) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# --- 3. LOAD MODELS ---
try:
    xgb = joblib.load(os.path.join(base_dir, "xgb_model.pkl"))
    scaler = joblib.load(os.path.join(base_dir, "scaler.pkl"))
    mlp = tf.keras.models.load_model(os.path.join(base_dir, "mlp_model.keras"))
    print("MODELS LOADED SUCCESSFULLY")
except Exception as e:
    print(f"ERROR LOADING MODELS: {e}")

# --- 4. PREDICTION LOGIC ---
class InputData(BaseModel):
    cement: float
    slag: float
    flyash: float
    water: float
    superplasticizer: float
    coarse: float
    fine: float
    age: float

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# REMOVED 'async' here to handle 1000+ requests faster (Thread Pooling)
@app.post("/predict")
def predict(data: InputData):
    features = [
        data.cement, data.slag, data.flyash, data.water,
        data.superplasticizer, data.coarse, data.fine, data.age
    ]
    
    X = np.array([features])
    Xs = scaler.transform(X)
    
    # Hybrid Prediction
    pred_log = (0.6 * xgb.predict(X)) + (0.4 * mlp.predict(Xs).flatten())
    strength = np.expm1(pred_log)[0]

    return {"strength": round(float(strength), 2)}

# --- 5. AUTOMATIC LAUNCHER ---
if __name__ == "__main__":
    def open_browser():
        time.sleep(2) # Give server 2 seconds to start
        webbrowser.open("http://127.0.0.1:8000")

    print("Starting App... Please wait for browser to open.")
    threading.Thread(target=open_browser).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)