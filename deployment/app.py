import numpy as np
import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model


#CẤU HÌNH ĐƯỜNG DẪN

MODEL_PATH = "../model/kidney_disease_model.h5"   
SCALER_PATH = "../model/scaler.pkl"               


# LOAD MODEL & SCALER
print("🔁 Đang load mô hình và scaler...")
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("✅ Đã load mô hình & scaler xong.")

# KHAI BÁO CÁC FEATURE
FEATURE_COLUMNS = [
    "age",
    "bp",
    "sg",
    "al",
    "su",
    "rbc",
    "pc",
    "pcc",
    "ba",
    "bgr",
    "bu",
    "sc",
    "sod",
    "pot",
    "hemo",
    "pcv",
    "wc",
    "rc",
    "htn",
    "dm",
    "cad",
    "appet",
    "pe",
    "ane",
]


# ĐỊNH NGHĨA SCHEMA INPUT

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

   
# KHỞI TẠO FASTAPI

app = FastAPI(
    title="CKD Prediction API",
    description="API dự đoán bệnh suy thận mạn tính (CKD) sử dụng mô hình ANN",
    version="1.0.0"
)

# 6. HÀM TIỆN ÍCH DỰ ĐOÁN

def predict_ckd(features: PatientFeatures):
    """
    Nhận input dạng PatientFeatures, trả về:
    - prediction: 0 (Not CKD), 1 (CKD)
    - probability: xác suất CKD
    """
    # Chuyển về list theo đúng thứ tự FEATURE_COLUMNS
    data_dict = features.dict()
    input_list = [data_dict[col] for col in FEATURE_COLUMNS]

    # Chuyển thành mảng 2D cho scaler & model
    X = np.array([input_list], dtype=float)

    # Chuẩn hóa giống lúc train
    X_scaled = scaler.transform(X)

    # Dự đoán xác suất CKD
    prob = float(model.predict(X_scaled)[0][0])

    # Ngưỡng 0.5 (bạn có thể thay bằng threshold tối ưu nếu đã tinh chỉnh)
    pred_label = 1 if prob >= 0.5 else 0

    return pred_label, prob


# ROUTES CỦA API
@app.get("/")
def root():
    return {
        "message": "CKD Prediction API is running.",
        "usage": "Gửi POST /predict với JSON chứa các đặc trưng bệnh nhân.",
        "example_endpoint": "/predict"
    }

@app.post("/predict")
def predict_endpoint(features: PatientFeatures):
    pred_label, prob = predict_ckd(features)

    label_str = "CKD" if pred_label == 1 else "Not CKD"

    return {
        "prediction": pred_label,
        "label": label_str,
        "probability_ckd": prob,
        "probability_not_ckd": 1 - prob
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
