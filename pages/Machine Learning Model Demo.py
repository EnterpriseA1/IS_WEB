import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="House Price Prediction Demo",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS to improve appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1565C0;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .prediction-result {
        background-color: #e8f5e9;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 1.2rem;
        font-weight: 600;
        color: #2E7D32;
        text-align: center;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
    }
    .feature-info {
        font-size: 0.9rem;
        color: #555;
        font-style: italic;
        margin-top: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Import required libraries
from pathlib import Path
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import seaborn as sns
import pandas as pd

# Main header with improved styling
st.markdown('<div class="main-header">🧠 Machine Learning Model Demo</div>', unsafe_allow_html=True)
st.write("Interactive demo of machine learning models for house price prediction.")

# Load models and scalers
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
        
# Function to predict house prices
def predict_house_price_rf(features):
    # Scale features
    features_scaled = scalerRF.transform([features])  
    price_pred = rf_model.predict(features_scaled)
    return price_pred[0]

def predict_house_price_knr(features):
    # Scale features
    features_scaled = scalerKnr.transform([features])  
    price_pred = Knr_model.predict(features_scaled)
    return price_pred[0]
    
def load_training_data():
    # Load training data
    X_train_path = Path.cwd()/"model_training"/"X_train.pkl"
    y_train_path = Path.cwd()/"model_training"/"y_train.pkl"
        
    with open(X_train_path, "rb") as file:
        X_train = pickle.load(file)
    with open(y_train_path, "rb") as file:
        y_train = pickle.load(file)
            
    return X_train, y_train

def calculate_knr_importance():
    X_train, y_train = load_training_data()
    
    # Scale the data
    X_train_scaled = scalerKnr.transform(X_train)
    
    # Calculate permutation importance
    result = permutation_importance(
        Knr_model, X_train_scaled, y_train, 
        n_repeats=10, random_state=42, n_jobs=-1
    )
    return result.importances_mean

# Function to plot feature importance
def plot_feature_importance(importances, feature_names, title, color_palette="viridis"):
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette=color_palette)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Features")
    return fig

# Enhanced UI for form inputs
st.markdown('<div class="sub-header">🏡 House Price Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.write("กรอกข้อมูลเกี่ยวกับบ้านเพื่อทำนายราคา")
st.markdown('</div>', unsafe_allow_html=True)

# Create two columns for the form inputs
col1, col2 = st.columns(2)

with col1:
    area = st.selectbox("Area (พื้นที่ที่อยู่อาศัย)", ["Urban", "Semiurban", "Rural"])
    coapplicant = st.selectbox("Coapplicant (ผู้ร่วมยื่นขอสินเชื่อ)", ["Yes", "No"])
    dependents = st.number_input("Dependents (จำนวนผู้ที่พึ่งพา)", min_value=0, max_value=10, value=0)
    income = st.number_input("Income (รายได้)", min_value=10000.0, max_value=1000000.0, value=50000.0, step=1000.0)
    loan_amount = st.number_input("Loan Amount (จำนวนเงินกู้)", min_value=10000.0, max_value=1000000.0, value=200000.0, step=10000.0)

with col2:
    property_age = st.number_input("Property Age (อายุทรัพย์สิน)", min_value=1, max_value=100, value=10)
    bedrooms = st.number_input("Bedrooms (จำนวนห้องนอน)", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms (จำนวนห้องน้ำ)", min_value=1, max_value=10, value=2)
    area_sqft = st.number_input("Area SqFt (ขนาดพื้นที่บ้าน ตร.ฟุต)", min_value=100, max_value=10000, value=1000)

# Map categorical values to numeric
area_mapping = {"Urban": 2, "Semiurban": 1, "Rural": 0}
coapplicant_mapping = {"Yes": 1, "No": 0}

area_value = area_mapping[area]
coapplicant_value = coapplicant_mapping[coapplicant]

feature_names = ["Area", "Coapplicant", "Dependents", "Income", "Loan Amount", "Property Age", "Bedrooms", "Bathrooms", "Area SqFt"]

# Create a two-column layout for the prediction buttons
st.markdown('<div class="section-header">Select a Model for Prediction</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    rf_button = st.button("🔍 Predict Price with Random Forest", use_container_width=True)

with col2:
    knr_button = st.button("🔍 Predict Price with KNN", use_container_width=True)

# Handle Random Forest prediction
if rf_button:
    input_features = np.array([area_value, coapplicant_value, dependents, income, loan_amount, property_age,
                             bedrooms, bathrooms, area_sqft])
    predicted_price = predict_house_price_rf(input_features)
    
    # Display prediction with improved styling
    st.markdown(f'<div class="prediction-result">🏠 ราคาที่คาดการณ์: {predicted_price:,.2f} บาท</div>', unsafe_allow_html=True)
    
    # Feature Importance section with better styling
    st.markdown('<div class="section-header">Feature Importance Analysis</div>', unsafe_allow_html=True)
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Create and display plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette="viridis")
    
    # Improve plot styling
    ax.set_title("Feature Importance in Random Forest Regression", fontsize=14, fontweight='bold')
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_ylabel("Features", fontsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Display the plot
    st.pyplot(fig)
    
    # Create a dataframe for importance values
    importance_df = pd.DataFrame({
        'Feature': np.array(feature_names)[indices],
        'Importance': importances[indices]
    })
    
    # Display the dataframe with better styling
    st.dataframe(importance_df, use_container_width=True)

# Handle KNN prediction
if knr_button:
    input_features = np.array([area_value, coapplicant_value, dependents, income, loan_amount, property_age,
                             bedrooms, bathrooms, area_sqft])
    predicted_price = predict_house_price_knr(input_features)
    
    # Display prediction with improved styling
    st.markdown(f'<div class="prediction-result">🏠 ราคาที่คาดการณ์: {predicted_price:,.2f} บาท</div>', unsafe_allow_html=True)
    
    # Feature Importance section with better styling
    st.markdown('<div class="section-header">Feature Importance Analysis</div>', unsafe_allow_html=True)
    
    # Show loading spinner while calculating importance
    with st.spinner("กำลังคำนวณค่าความสำคัญของคุณลักษณะ... กรุณารอสักครู่"):
        importances = calculate_knr_importance()
        
        # Create and display plot with improved styling
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        indices = np.argsort(importances)[::-1]
        
        bars = sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette="viridis", ax=ax)
        
        # Improve plot styling
        ax.set_title("Feature Importance in K-Nearest Neighbors Regression", fontsize=14, fontweight='bold')
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_ylabel("Features", fontsize=12)
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        # Display the plot
        st.pyplot(fig)
        
        # Create a dataframe for importance values
        importance_df = pd.DataFrame({
            'Feature': np.array(feature_names)[indices],
            'Importance': importances[indices]
        })
        
        # Display the dataframe with better styling
        st.dataframe(importance_df, use_container_width=True)

# Add a footer
st.markdown("""---""")
st.markdown("""
<div style="text-align: center; color: #666;">
    House Price Prediction - Machine Learning Demo © 2025
</div>
""", unsafe_allow_html=True)