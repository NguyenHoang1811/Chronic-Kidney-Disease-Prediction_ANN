import numpy as np
import joblib
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from pathlib import Path

# CẤU HÌNH ĐƯỜNG DẪN
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "model" / "best_kidney_model.h5"
SCALER_PATH = BASE_DIR.parent / "model" / "scaler.pkl"

# LOAD MODEL & SCALER
print("🔁 Đang load mô hình và scaler...")
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("✅ Đã load mô hình & scaler xong.")

# FEATURE ORDER
FEATURE_COLUMNS = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
    "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc",
    "htn", "dm", "cad", "appet", "pe", "ane"
]

# SCHEMA INPUT
class PatientFeatures(BaseModel):
    age: float
    bp: float
    sg: float
    al: float
    su: float
    rbc: int
    pc: int
    pcc: int
    ba: int
    bgr: float
    bu: float
    sc: float
    sod: float
    pot: float
    hemo: float
    pcv: float
    wc: float
    rc: float
    htn: int
    dm: int
    cad: int
    appet: int
    pe: int
    ane: int

# FASTAPI APP
app = FastAPI(
    title="CKD Prediction API",
    description="API dự đoán bệnh suy thận mạn",
    version="1.0.0"
)

# PREDICT FUNCTION
def predict_ckd(features: PatientFeatures):
    data = features.dict()
    X = np.array([[data[col] for col in FEATURE_COLUMNS]], dtype=float)
    X_scaled = scaler.transform(X)
    prob = float(model.predict(X_scaled)[0][0])
    label = 1 if prob >= 0.5 else 0
    return label, prob

# ROUTES - SỬA LỖI Ở ĐÂY
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Phục vụ trang chủ"""
    html_path = BASE_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    else:
        return HTMLResponse(content="""
        <html>
            <head><title>CKD Prediction System</title></head>
            <body>
                <h1>Hệ thống chẩn đoán CKD</h1>
                <p>Giao diện đang được tải...</p>
                <p>Vui lòng truy cập:</p>
                <ul>
                    <li><a href="/api/docs">API Documentation</a></li>
                    <li><a href="/api/health">Health Check</a></li>
                </ul>
            </body>
        </html>
        """)

@app.get("/api/health")
async def health_check():
    """Kiểm tra tình trạng API"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "service": "CKD Prediction API"
    }

@app.post("/predict")
async def predict_api(features: PatientFeatures):
    """Endpoint dự đoán CKD"""
    try:
        label, prob = predict_ckd(features)
        return {
            "success": True,
            "prediction": label,
            "label": "CKD" if label == 1 else "Not CKD",
            "probability_ckd": prob,
            "probability_not_ckd": 1 - prob,
            "message": "Dự đoán thành công"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "message": "Có lỗi xảy ra khi dự đoán"
            }
        )

if __name__ == "__main__":
    import uvicorn
    print("Khởi động CKD Prediction System...")
    print("Truy cập: http://localhost:8000")
    print("API Docs: http://localhost:8000/api/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)