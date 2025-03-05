import streamlit as st

st.title("🧠 Machine Learning Model Demo")
st.write("Interactive demo of machine learning models.")

from pathlib import Path
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# ฟังก์ชันทำนายราคาบ้าน
def predict_house_price(features):
    # โหลดโมเดลที่เทรนไว้และโหลดตัวปรับมาตรฐาน (Scaler) ใหม่ทุกครั้ง
    model_path = Path.cwd()/"model_training"/ "modelSVM.pkl"
    scaler_path = Path.cwd()/"model_training" / "scalerSVM.pkl"

    with open(model_path, "rb") as file:
        svr_model = pickle.load(file)

    with open(scaler_path, "rb") as file:
        scaler = pickle.load(file)

    # ปรับค่าฟีเจอร์
    features_scaled = scaler.transform([features])  
    price_pred = svr_model.predict(features_scaled)
    return price_pred[0]

# UI ของ Streamlit
st.title("🏡 House Price Prediction")
st.write("กรอกข้อมูลเกี่ยวกับบ้านเพื่อทำนายราคา")

# อินพุตฟอร์มสำหรับรับข้อมูล
area = st.selectbox("Area (พื้นที่ที่อยู่อาศัย)", ["Urban", "Semiurban", "Rural"])
coapplicant = st.selectbox("Coapplicant (ผู้ร่วมยื่นขอสินเชื่อ)", ["Yes", "No"])
dependents = st.number_input("Dependents (จำนวนผู้ที่พึ่งพา)", min_value=0, max_value=10, value=0)
income = st.number_input("Income (รายได้)", min_value=0.0, max_value=1000000.0, value=50000.0)
loan_amount = st.number_input("Loan Amount (จำนวนเงินกู้)", min_value=1000.0, max_value=1000000.0, value=200000.0)
property_age = st.number_input("Property Age (อายุทรัพย์สิน)", min_value=0, max_value=100, value=10)
bedrooms = st.number_input("Bedrooms (จำนวนห้องนอน)", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Bathrooms (จำนวนห้องน้ำ)", min_value=1, max_value=10, value=2)
area_sqft = st.number_input("Area SqFt (ขนาดพื้นที่บ้าน ตร.ฟุต)", min_value=100, max_value=10000, value=1000)

# แปลงค่า categorical เป็นตัวเลข

area_mapping = {"Urban": 2, "Semiurban": 1, "Rural": 0}
coapplicant_mapping = {"Yes": 1, "No": 0}

# แปลงค่าเป็นตัวเลข
area_value = area_mapping[area]
coapplicant_value = coapplicant_mapping[coapplicant]


# กดปุ่มเพื่อทำนาย
if st.button("🔍 Predict Price"):
    input_features = np.array([area_value, coapplicant_value, dependents, income, loan_amount, property_age,
                               bedrooms, bathrooms, area_sqft])
    predicted_price = predict_house_price(input_features)
    st.success(f"🏠 ราคาที่คาดการณ์: {predicted_price:,.2f} บาท")
    
    