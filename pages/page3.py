import streamlit as st

st.title("🧠 Machine Learning Model Demo")
st.write("Interactive demo of machine learning models.")

from pathlib import Path
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import seaborn as sns

model_path_RF = Path.cwd()/"model_training"/ "modelRF.pkl"
scaler_path_RF = Path.cwd()/"model_training" / "scalerRF.pkl"

model_path_knr = Path.cwd()/"model_training"/ "modelKnr.pkl"
scaler_path_knr = Path.cwd()/"model_training" / "scalerKNR.pkl"
with open(model_path_RF, "rb") as file:
    rf_model = pickle.load(file)

with open(scaler_path_RF, "rb") as file:
    scalerRF = pickle.load(file)
    
with open(model_path_knr, 'rb') as file:
    Knr_model = pickle.load(file)

with open(scaler_path_knr, 'rb') as file:
    scalerKnr = pickle.load(file)
        
# ฟังก์ชันทำนายราคาบ้าน
def predict_house_price_rf(features):
    # ปรับค่าฟีเจอร์
    features_scaled = scalerRF.transform([features])  
    price_pred = rf_model.predict(features_scaled)
    return price_pred[0]
def predict_house_price_knr(features):
    # ปรับค่าฟีเจอร์
    features_scaled = scalerKnr.transform([features])  
    price_pred = Knr_model.predict(features_scaled)
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
if st.button("🔍 Predict Price Random forest"):
    input_features = np.array([area_value, coapplicant_value, dependents, income, loan_amount, property_age,
                               bedrooms, bathrooms, area_sqft])
    predicted_price = predict_house_price_rf(input_features)
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
if st.button("🔍 Predict Price KNR"):
    input_features = np.array([area_value, coapplicant_value, dependents, income, loan_amount, property_age,
                               bedrooms, bathrooms, area_sqft])
    predicted_price = predict_house_price_knr(input_features)
    st.success(f"🏠 ราคาที่คาดการณ์: {predicted_price:,.2f} บาท")
    
   
    
    
    