import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ===============================
# LOAD MODEL & SCALER
# ===============================
MODEL_PATH = "../model/best_kidney_model.h5"
SCALER_PATH = "../model/scaler.pkl"

@st.cache_resource
def load_resources():
    
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

try:
    model, scaler = load_resources()
except Exception as e:
    st.error(f"Lỗi khi load model hoặc scaler: {e}")
    st.stop()

# ===============================
# GIAO DIỆN
# ===============================
st.set_page_config(page_title="CKD Prediction", layout="centered")
st.title("DỰ ĐOÁN NGUY CƠ MẮC SUY THẬN MẠN (CKD)")
st.write("Vui lòng nhập đầy đủ các thông số bệnh nhân.")

st.divider()

# ===============================
# FORM NHẬP LIỆU
# ===============================
with st.form("ckd_form"):
    age = st.number_input("Tuổi", min_value=0, max_value=120, value=None)
    bp = st.number_input("Huyết áp (bp)", min_value=0, max_value=200, value=None)
    sg = st.selectbox("Tỷ trọng nước tiểu (sg)", ["", 1.005, 1.010, 1.015, 1.020, 1.025])
    al = st.selectbox("Albumin (al)", ["", 0, 1, 2, 3, 4, 5])
    su = st.selectbox("Đường niệu (su)", ["", 0, 1, 2, 3, 4, 5])

    rbc = st.selectbox("Hồng cầu niệu (rbc)", ["", "Normal", "Abnormal"])
    pc = st.selectbox("Tế bào mủ (pc)", ["", "Normal", "Abnormal"])
    pcc = st.selectbox("Mủ kết tụ (pcc)", ["", "No", "Yes"])
    ba = st.selectbox("Vi khuẩn (ba)", ["", "No", "Yes"])

    bgr = st.number_input("Đường huyết ngẫu nhiên (bgr)", value=None)
    bu = st.number_input("Ure máu (bu)", value=None)
    sc = st.number_input("Creatinine huyết thanh (sc)", value=None)
    sod = st.number_input("Natri (sod)", value=None)
    pot = st.number_input("Kali (pot)", value=None)

    hemo = st.number_input("Hemoglobin (hemo)", value=None)
    pcv = st.number_input("Hematocrit (pcv)", value=None)
    wc = st.number_input("Bạch cầu (wc)", value=None)
    rc = st.number_input("Hồng cầu (rc)", value=None)

    htn = st.selectbox("Tăng huyết áp", ["", "No", "Yes"])
    dm = st.selectbox("Tiểu đường", ["", "No", "Yes"])
    cad = st.selectbox("Bệnh tim mạch", ["", "No", "Yes"])
    appet = st.selectbox("Cảm giác ăn uống", ["", "Poor", "Good"])
    pe = st.selectbox("Phù", ["", "No", "Yes"])
    ane = st.selectbox("Thiếu máu", ["", "No", "Yes"])

    submit = st.form_submit_button("DỰ ĐOÁN")

# ===============================
# XỬ LÝ & DỰ ĐOÁN
# ===============================
if submit:
    # Kiểm tra nhập đủ dữ liệu
    if "" in [
        sg, al, su, rbc, pc, pcc, ba,
        htn, dm, cad, appet, pe, ane
    ] or None in [
        age, bp, bgr, bu, sc, sod, pot,
        hemo, pcv, wc, rc
    ]:
        st.warning("Vui lòng nhập đầy đủ tất cả các trường dữ liệu!")
        st.stop()

    # Encode categorical → số
    def yes_no(x): return 1 if x == "Yes" else 0
    def normal_abnormal(x): return 1 if x == "Abnormal" else 0
    def appet_map(x): return 1 if x == "Good" else 0

    input_data = np.array([[  
        age, bp, float(sg), float(al), float(su),
        normal_abnormal(rbc),
        normal_abnormal(pc),
        yes_no(pcc),
        yes_no(ba),
        bgr, bu, sc, sod, pot,
        hemo, pcv, wc, rc,
        yes_no(htn),
        yes_no(dm),
        yes_no(cad),
        appet_map(appet),
        yes_no(pe),
        yes_no(ane)
    ]])

    # Chuẩn hóa
    input_scaled = scaler.transform(input_data)

    # Dự đoán
    prob = model.predict(input_scaled)[0][0]

    result = (
        "CÓ NGUY CƠ MẮC SUY THẬN"
        if prob >= 0.5
        else "KHÔNG CÓ NGUY CƠ MẮC SUY THẬN"
    )

    st.divider()
    st.subheader("KẾT QUẢ DỰ ĐOÁN")
    st.metric("Kết luận", result)
    st.write(f"Xác suất mắc CKD: **{prob:.2%}**")
