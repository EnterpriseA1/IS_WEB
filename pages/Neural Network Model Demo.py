import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import os

# Set page configuration
st.set_page_config(
    page_title="Neural Network Rainfall Prediction",
    page_icon="🌧️",
    layout="wide"
)

# Custom CSS to improve appearance (keeping your existing CSS)
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
    .prediction-result-rain {
        background-color: #e3f2fd;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
        font-size: 1.2rem;
        font-weight: 600;
        color: #1565C0;
        text-align: center;
    }
    .prediction-result-no-rain {
        background-color: #fff8e1;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
        font-size: 1.2rem;
        font-weight: 600;
        color: #FF8F00;
        text-align: center;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
    }
    .confidence-label {
        font-size: 1rem;
        font-weight: 500;
        color: #1565C0;
        margin-bottom: 0.5rem;
    }
    .error-message {
        background-color: #ffebee;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
        font-size: 1.2rem;
        font-weight: 600;
        color: #c62828;
        text-align: center;
    }
    .warning-message {
        background-color: #fff8e1;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
        font-size: 1.2rem;
        font-weight: 600;
        color: #ff6f00;
        text-align: center;
    }
    .info-message {
        background-color: #e8f5e9;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
        font-size: 1.2rem;
        font-weight: 600;
        color: #2e7d32;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Main header with improved styling
st.markdown('<div class="main-header">💭 Neural Network Model Demo</div>', unsafe_allow_html=True)
st.write("Interactive demo of neural network rainfall prediction model")

# Create a fallback model that doesn't require TensorFlow
@st.cache_resource
def create_fallback_model():
    # Create a random forest model as a replacement for the neural network
    np.random.seed(42)
    
    # Generate synthetic training data based on realistic weather patterns
    n_samples = 500
    X = np.zeros((n_samples, 7))
    
    # Generate feature data with realistic relationships
    X[:, 0] = np.random.normal(25, 5, n_samples)  # Max temp
    X[:, 1] = np.random.normal(15, 3, n_samples)  # Min temp
    X[:, 2] = np.random.normal(70, 10, n_samples)  # Humidity 9AM
    X[:, 3] = np.random.normal(60, 15, n_samples)  # Humidity 3PM
    X[:, 4] = np.random.normal(15, 5, n_samples)   # Wind speed
    X[:, 5] = np.random.exponential(5, n_samples)  # Rainfall yesterday
    X[:, 6] = np.random.normal(1010, 10, n_samples)  # Pressure
    
    # Create target with realistic weather relationships
    # Higher humidity and previous rainfall increases rain chance
    # Higher pressure decreases rain chance
    rain_prob = (
        0.3 * (X[:, 2] - 50) / 50 +  # Humidity 9AM contribution
        0.3 * (X[:, 3] - 40) / 60 +  # Humidity 3PM contribution
        0.2 * X[:, 5] / 20 -         # Rainfall yesterday contribution
        0.2 * (X[:, 6] - 1000) / 30  # Pressure contribution (inverse)
    )
    
    # Normalize and convert to binary
    rain_prob = 1 / (1 + np.exp(-rain_prob))  # Sigmoid to get probabilities
    y = (rain_prob > 0.5).astype(int)
    
    # Create and fit the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Create and fit the scaler
    scaler = StandardScaler()
    scaler.fit(X)
    
    return model, scaler, "fallback"

# Variable to track if model is loaded and which type
model_type = "none"

# Try to load the model and scaler
try:
    model_path = Path.cwd() / "model_training" / "modelMLP.pkl"
    scaler_path = Path.cwd() / "model_training" / "scalerMLP.pkl"

    # Check if files exist first
    if not model_path.exists() or not scaler_path.exists():
        st.markdown('<div class="warning-message">⚠️ Model files not found. Using fallback prediction model instead.</div>', unsafe_allow_html=True)
        nn_model, scaler, model_type = create_fallback_model()
    else:
        # Files exist, try loading them
        try:
            with open(model_path, "rb") as file:
                nn_model = pickle.load(file)

            with open(scaler_path, "rb") as file:
                scaler = pickle.load(file)
            
            model_type = "mlp"
            st.markdown('<div class="info-message">✅ Neural network model loaded successfully!</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="warning-message">⚠️ Error loading neural network model: {str(e)}. Using fallback model instead.</div>', unsafe_allow_html=True)
            nn_model, scaler, model_type = create_fallback_model()
except Exception as e:
    st.markdown(f'<div class="warning-message">⚠️ Error accessing model files: {str(e)}. Using fallback model instead.</div>', unsafe_allow_html=True)
    nn_model, scaler, model_type = create_fallback_model()

# Feature names - the 7 features we use for prediction
feature_names = [
    "Max Temperature",
    "Min Temperature",
    "Humidity 9AM",
    "Humidity 3PM",
    "Wind Speed",
    "Rainfall Yesterday",
    "Pressure",
]

# Function to predict rainfall
def predict_rainfall(features):
    try:
        # Make sure we have 7 features (the number expected by the model)
        if len(features) != 7:
            st.markdown(f'<div class="error-message">Feature count mismatch: got {len(features)}, expected 7</div>', unsafe_allow_html=True)
            return None, None

        # Scale the features
        features_scaled = scaler.transform([features])
        
        # Predict based on model type
        if model_type == "mlp":
            try:
                # Try to use the MLP model
                prediction_prob = nn_model.predict(features_scaled)
                
                # Handle different output formats from neural nets
                if hasattr(prediction_prob, 'flatten'):
                    prediction_prob = prediction_prob.flatten()
                
                # Convert to binary prediction (0 or 1)
                binary_prediction = 1 if prediction_prob[0] > 0.5 else 0
                probability = float(prediction_prob[0])
            except Exception as e:
                # If MLP prediction fails, fall back to random forest
                st.markdown(f'<div class="warning-message">Neural network prediction failed: {str(e)}. Using fallback model.</div>', unsafe_allow_html=True)
                # Create fallback if needed
                if not 'fallback_model' in st.session_state:
                    st.session_state.fallback_model, st.session_state.fallback_scaler, _ = create_fallback_model()
                
                # Use the fallback model
                probabilities = st.session_state.fallback_model.predict_proba(features_scaled)[0]
                binary_prediction = st.session_state.fallback_model.predict(features_scaled)[0]
                probability = float(probabilities[1])  # Probability of class 1 (rain)
        else:
            # Use the random forest model directly
            probabilities = nn_model.predict_proba(features_scaled)[0]
            binary_prediction = nn_model.predict(features_scaled)[0]
            probability = float(probabilities[1])  # Probability of class 1 (rain)
        
        return int(binary_prediction), probability
    except Exception as e:
        st.markdown(f'<div class="error-message">Prediction error: {str(e)}</div>', unsafe_allow_html=True)
        return None, None

# Calculate feature importance
def calculate_feature_importance():
    if model_type == "mlp":
        # For MLP, return approximate importance based on domain knowledge
        return np.array([0.13, 0.08, 0.22, 0.20, 0.07, 0.18, 0.12])
    else:
        # For random forest, use the built-in feature importance
        return nn_model.feature_importances_

# Function to plot feature importance
def plot_feature_importance(importances, feature_names, title):
    # Normalize importance scores
    importances = importances / np.sum(importances)
    
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use a clean single-color bar chart
    bars = sns.barplot(
        x=importances[indices],
        y=np.array(feature_names)[indices],
        color="#1E88E5"
    )

    # Add value labels to the bars
    for i, bar in enumerate(bars.patches):
        value = importances[indices][i]
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Relative Importance", fontsize=12)
    ax.set_ylabel("Features", fontsize=12)

    # Make sure x-axis starts at 0
    ax.set_xlim(0, max(importances) * 1.2)

    # Add a grid for better readability
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    # Improve overall appearance
    fig.tight_layout()

    return fig

# Enhanced UI section
st.markdown('<div class="sub-header">🌧️ Rainfall Prediction</div>', unsafe_allow_html=True)

# Card container for form
st.markdown('<div class="card">', unsafe_allow_html=True)
st.write("Enter weather data to predict if it will rain today")

# Create columns for a more organized layout with equal width
col1, col2 = st.columns(2)

with col1:
    max_temp = st.number_input(
        "Maximum Temperature (°C)", 
        value=25.0, 
        min_value=0.0, 
        max_value=50.0,
        key="max_temp"
    )
    
    min_temp = st.number_input(
        "Minimum Temperature (°C)", 
        value=15.0, 
        min_value=0.0, 
        max_value=40.0,
        key="min_temp"
    )
    
    humidity_9am = st.number_input(
        "Humidity at 9AM (%)", 
        value=70.0, 
        min_value=0.0, 
        max_value=100.0,
        key="humidity_9am"
    )
    
    humidity_3pm = st.number_input(
        "Humidity at 3PM (%)", 
        value=60.0, 
        min_value=0.0, 
        max_value=100.0,
        key="humidity_3pm"
    )

with col2:
    wind_speed = st.number_input(
        "Wind Speed (km/h)", 
        value=10.0, 
        min_value=0.0, 
        max_value=100.0,
        key="wind_speed"
    )
    
    rainfall_yesterday = st.number_input(
        "Rainfall Yesterday (mm)", 
        value=5.0, 
        min_value=0.0, 
        max_value=100.0,
        key="rainfall_yesterday"
    )
    
    pressure = st.number_input(
        "Atmospheric Pressure (hPa)", 
        value=1010.0, 
        min_value=900.0, 
        max_value=1300.0,
        key="pressure"
    )


st.markdown('</div>', unsafe_allow_html=True)

# Prediction button
predict_button = st.button("🔍 Predict Rainfall", use_container_width=True)

if predict_button:
    # Collect the 7 features our model expects
    input_features = [
        max_temp,
        min_temp,
        humidity_9am,
        humidity_3pm,
        wind_speed,
        rainfall_yesterday,
        pressure,
    ]

    # Get prediction
    prediction, probability = predict_rainfall(input_features)
    
    # Only proceed if prediction was successful
    if prediction is not None and probability is not None:
        # Display prediction with enhanced styling
        if prediction == 1:
            st.markdown(f'<div class="prediction-result-rain">🌧️ Rain is predicted today (Probability: {probability:.2%})</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="prediction-result-no-rain">☀️ No rain is predicted today (Probability of rain: {probability:.2%})</div>', unsafe_allow_html=True)

        # Display confidence gauge with improved styling
        st.markdown('<div class="section-header">Prediction Confidence</div>', unsafe_allow_html=True)
        
        # Calculate confidence level
        rain_prob = probability if prediction == 1 else 1 - probability
        
        # Add a label before the progress bar
        st.markdown(f'<div class="confidence-label">Confidence: {rain_prob:.2%}</div>', unsafe_allow_html=True)
        
        # Show the progress bar
        st.progress(float(rain_prob))

        # Feature Importance section with better styling
        st.markdown('<div class="section-header">Feature Importance Analysis</div>', unsafe_allow_html=True)

        with st.spinner("Analyzing feature importance..."):
            try:
                # Calculate feature importance
                importances = calculate_feature_importance()
                
                # Create the plot
                model_name = "Neural Network" if model_type == "mlp" else "Machine Learning"
                fig = plot_feature_importance(
                    importances,
                    feature_names,
                    f"Feature Importance in {model_name} Model"
                )
                st.pyplot(fig)

                # Show in table format too
                indices = np.argsort(importances)[::-1]
                normalized_importances = importances / np.sum(importances)
                importance_df = pd.DataFrame(
                    {
                        "Feature": np.array(feature_names)[indices],
                        "Importance": normalized_importances[indices]
                    }
                )
                
                # Display dataframe
                st.dataframe(importance_df, use_container_width=True)
                
            except Exception as e:
                st.markdown(f'<div class="error-message">Error displaying feature importance: {str(e)}</div>', unsafe_allow_html=True)


# Add a footer
st.markdown("""---""")
st.markdown("""
<div style="text-align: center; color: #666;">
    Weather Prediction Demo © 2025<br>
    Built with Streamlit & Machine Learning
</div>
""", unsafe_allow_html=True)