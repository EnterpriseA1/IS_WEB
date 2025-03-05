import streamlit as st

st.title("🧠 Machine Learning Model Demo")
st.write("Interactive demo of machine learning models.")

from pathlib import Path
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.pyplot as plt


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
    
    # PDP: Partial Dependence Plot สำหรับทุกฟีเจอร์
    st.write("### Partial Dependence Plot (PDP) for All Features")

    # สร้างข้อมูลจำลองสำหรับการทำนาย
    feature_names = ["Area", "Coapplicant", "Dependents", "Income", "Loan Amount", "Property Age", "Bedrooms", "Bathrooms", "Area SqFt"]
    
    # สร้างลิสต์เพื่อเก็บผลลัพธ์ของ PDP
    pdp_values = {feature: [] for feature in feature_names}
    feature_ranges = {
        "Area": np.linspace(0, 2, 100),
        "Coapplicant": np.linspace(0, 1, 100),
        "Dependents": np.linspace(0, 10, 100),
        "Income": np.linspace(0, 1000000, 100),
        "Loan Amount": np.linspace(1000, 1000000, 100),
        "Property Age": np.linspace(0, 100, 100),
        "Bedrooms": np.linspace(1, 10, 100),
        "Bathrooms": np.linspace(1, 10, 100),
        "Area SqFt": np.linspace(100, 10000, 100)
    }

    # คำนวณ PDP สำหรับทุกฟีเจอร์
    for feature in feature_names:
        feature_index = feature_names.index(feature)
        feature_range = feature_ranges[feature]
        
        for value in feature_range:
            temp_features = input_features.copy()
            temp_features[feature_index] = value  # เปลี่ยนแปลงค่าฟีเจอร์ที่เลือก
            price = predict_house_price(temp_features)
            pdp_values[feature].append(price)
    
    # Plot PDP สำหรับทุกฟีเจอร์
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for feature in feature_names:
        ax.plot(feature_ranges[feature], pdp_values[feature], label=feature)
    
    ax.set_title("Partial Dependence Plot for All Features")
    ax.set_xlabel("Feature Value")
    ax.set_ylabel("Predicted Price (Baht)")
    ax.legend(loc="upper left")
    
    st.pyplot(fig)
    
    