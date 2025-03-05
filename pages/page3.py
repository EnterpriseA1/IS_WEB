import streamlit as st

st.title("🧠 Machine Learning Model Demo")
st.write("Interactive demo of machine learning models.")

from pathlib import Path
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.pyplot as plt

import seaborn as sns
# โหลดโมเดลที่เทรนไว้และโหลดตัวปรับมาตรฐาน (Scaler) ใหม่ทุกครั้ง
model_path = Path.cwd()/"model_training"/ "modelRF.pkl"
scaler_path = Path.cwd()/"model_training" / "scalerRF.pkl"

with open(model_path, "rb") as file:
    rf_model = pickle.load(file)

with open(scaler_path, "rb") as file:
    scaler = pickle.load(file)
        
# ฟังก์ชันทำนายราคาบ้าน
def predict_house_price(features):
    # ปรับค่าฟีเจอร์
    features_scaled = scaler.transform([features])  
    price_pred = rf_model.predict(features_scaled)
    return price_pred[0]
    

    

# UI ของ Streamlit
st.title("🏡 House Price Prediction")
st.write("กรอกข้อมูลเกี่ยวกับบ้านเพื่อทำนายราคา")

# อินพุตฟอร์มสำหรับรับข้อมูล
area = st.selectbox("Area (พื้นที่ที่อยู่อาศัย)", ["Urban", "Semiurban", "Rural"])
coapplicant = st.selectbox("Coapplicant (ผู้ร่วมยื่นขอสินเชื่อ)", ["Yes", "No"])
dependents = st.number_input("Dependents (จำนวนผู้ที่พึ่งพา)", value=0)
income = st.number_input("Income (รายได้)", value=50000.0)
loan_amount = st.number_input("Loan Amount (จำนวนเงินกู้)" ,value=200000.0)
property_age = st.number_input("Property Age (อายุทรัพย์สิน)", value=10)
bedrooms = st.number_input("Bedrooms (จำนวนห้องนอน)",  value=3)
bathrooms = st.number_input("Bathrooms (จำนวนห้องน้ำ)",  value=2)
area_sqft = st.number_input("Area SqFt (ขนาดพื้นที่บ้าน ตร.ฟุต)", value=1000)

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
    
    # Feature Importance Plot
    st.write("### Feature Importance")
    feature_names = ["Area", "Coapplicant", "Dependents", "Income", "Loan Amount", "Property Age", "Bedrooms", "Bathrooms", "Area SqFt"]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette="viridis")
    ax.set_title("Feature Importance in Random Forest Regression")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Features")
    st.pyplot(fig)
    
    
    
    