import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Neural Network Explanation", layout="wide")

def main():
    st.title("🔶 Neural Network Explanation")
    
    # Create tabs
    tabs = st.tabs(["Data Preparation", "Model Training", "Training & Results"])
    
    with tabs[0]:
        st.header("Data Preparation")
        
        # Dataset Overview
        st.subheader("Dataset Overview: rainfall_data_improved.csv")
        
        # Show sample data
        data = {
            "MaxTemperature": [20.59, 20.13, 39.21, 38.03, 29.45],
            "MinTemperature": [11.08, 8.68, 22.34, 8.40, 18.13],
            "Humidity9AM": [85.82, 76.18, 94.98, 43.63, 55.21],
            "Humidity3PM": [76.19, 66.74, 101.42, 36.90, 62.31],
            "WindSpeed": [14.08, 8.77, 4.46, 14.09, 15.84],
            "RainfallYesterday": [19.36, 0.20, 17.99, 4.85, 3.27],
            "Pressure": [1020.94, 998.75, 987.50, 1012.74, 998.94],
            "RainToday": [1.0, 0.0, 1.0, 0.0, 0.0]
        }
        df_example = pd.DataFrame(data)
        st.dataframe(df_example)
        
        st.write("- **Target Variable**: RainToday (1.0 = Rain, 0.0 = No Rain)")
        st.write("- **Total Samples**: 667 records after cleaning")
        
        # Data Preparation Steps
        st.subheader("My Data Preprocessing Steps")
        
        st.markdown("### 1. Loading and Exploring the Dataset")
        st.code("""
# Import dataset
df = pd.read_csv("../Dataset/rainfall_data_improved.csv")

# Display dataset info
df.shape  # Check dimensions
df.head() # View first few rows
""", language="python")
        
        st.markdown("### 2. Handling Missing Values")
        st.code("""
# Drop rows with missing values
df.dropna(inplace=True)
# Number of rows after cleaning: 667
""", language="python")
        
        st.write("""
        I handled missing values by removing rows with any null values. This ensures
        that the model is trained only on complete data records.
        """)
        
        st.markdown("### 3. Feature and Target Selection")
        st.code("""
# แยกฟีเจอร์และ target
X = df.drop('RainToday', axis=1).values  # ฟีเจอร์ทั้งหมด (ไม่รวม RainToday)
y = df['RainToday'].values  # Target: RainToday (ค่าฝน)
""", language="python")
        
        st.write("""
        I used all available weather metrics as features:
        - MaxTemperature: Maximum daily temperature
        - MinTemperature: Minimum daily temperature
        - Humidity9AM: Morning humidity percentage
        - Humidity3PM: Afternoon humidity percentage
        - WindSpeed: Wind speed in km/h
        - RainfallYesterday: Previous day's rainfall amount
        - Pressure: Atmospheric pressure
        
        The target variable 'RainToday' is binary (1.0 = Rain, 0.0 = No Rain).
        """)
        
        st.markdown("### 4. Feature Scaling with StandardScaler")
        st.code("""
# Standardize the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
""", language="python")
        
        st.write("""
        Neural networks perform better when input features are on a similar scale. I used StandardScaler 
        to normalize each feature to have a mean of 0 and standard deviation of 1.
        
        This is particularly important for my dataset because:
        - Pressure values (around 1000) are much larger than temperature values (20-40)
        - Wind speed and rainfall values vary on different scales
        - Standardization helps the neural network converge faster during training
        """)
        
        # Create a simplified visualization showing before/after scaling
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        
        # Create some example data
        before_scaling = np.array([
            [25, 12, 70, 65, 10, 5, 1000],
            [30, 15, 80, 75, 15, 8, 1010],
            [22, 10, 60, 55, 8, 2, 990]
        ])
        
        # Before scaling - show as bar chart
        features = ['MaxTemp', 'MinTemp', 'Hum9AM', 'Hum3PM', 'Wind', 'Rain', 'Press']
        ax[0].bar(features, before_scaling[0])
        ax[0].set_title('Before Standardization')
        ax[0].set_ylabel('Value')
        ax[0].tick_params(axis='x', rotation=45)
        
        # After scaling
        after_scaling = np.array([
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ])
        ax[1].bar(features, after_scaling[0])
        ax[1].set_title('After Standardization')
        ax[1].set_ylabel('Standardized Value')
        ax[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("### 5. Train-Test Split")
        st.code("""
# Split data into 80% training and 20% testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, 
    y, 
    test_size=0.2,   # 20% for testing
    random_state=42  # For reproducibility
)
""", language="python")
        
        st.write("""
        I split the dataset into training (80%) and testing (20%) sets using scikit-learn's train_test_split function.
        
        The random_state=42 parameter ensures that the split is reproducible, which is important for:
        - Consistent results when rerunning the code
        - Fair comparison between different model configurations
        - Debugging and validating the model's performance
        """)
        
        # Simple pie chart for train/test split
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie([80, 20], labels=['Training (80%)', 'Testing (20%)'], 
              autopct='%1.1f%%', 
              startangle=90, 
              colors=['#4CAF50', '#2196F3'])
        ax.axis('equal')
        st.pyplot(fig)
    
    with tabs[1]:
        # Simple visual representation of the network
        cols = st.columns([1, 3, 1])
        with cols[1]:
            st.write("""
            """)
        
        # Model Building Code - exactly as in MLP.ipynb
        st.subheader("Building the Model in Code")
        
        st.markdown("### 1. Creating the Model Object")
        st.code("""
# สร้างโมเดล MLP
model = Sequential()
""", language="python")
        
        st.markdown("### 2. Adding Hidden Layers")
        st.code("""
# เพิ่ม Hidden layers
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))  # ขนาดของ input_dim ขึ้นอยู่กับจำนวนฟีเจอร์
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
""", language="python")
        
        st.markdown("### 3. Adding Output Layer")
        st.code("""
# เพิ่ม Output layer (ค่าผลลัพธ์เป็นตัวเลขเดียว)
model.add(Dense(1,activation="sigmoid"))  # สำหรับ regression ให้ใช้จำนวน output = 1
""", language="python")
        
        # Explain each layer
        st.subheader("Understanding the Layers")
        st.write("""
        1. **Input Layer (7 neurons)**: Receives the weather data features
           
        2. **Hidden Layers (128 → 64 → 32 → 16 → 8 neurons)**: 
           - Each neuron applies a ReLU activation function
           - The decreasing pattern helps the network learn hierarchical features
           
        3. **Output Layer (1 neuron)**:
           - Uses a sigmoid activation function
           - Outputs a value between 0 and 1 (probability of rain)
           - Predictions above 0.5 are classified as "Rain Today"
        """)
    
    with tabs[2]:
        st.header("Training & Results")
        
        # Training section following exactly the MLP.ipynb
        st.subheader("Training the Model")
        
        st.markdown("### 1. Compiling the Model")
        st.code("""
# คอมไพล์โมเดล
model.compile(optimizer=Adam(), loss='mean_squared_error')
""", language="python")
        
        st.markdown("### 2. Training the Model")
        st.code("""
# ฝึกโมเดล
history = model.fit(X_train, y_train, epochs=25, batch_size=16, validation_data=(X_test, y_test))
""", language="python")
        
        # Display actual training output from notebook
        
        st.markdown("### 3. Making Predictions")
        st.code("""
# ทำนายผล
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)
""", language="python")
        
        # Training progress visualization simplified
        st.subheader("Training Progress")
        
        # Create simplified training history visualization
        epochs = range(1, 26)
        train_loss = [0.23, 0.10, 0.05, 0.04, 0.04, 0.04, 0.04, 0.03, 0.03, 0.02, 
                     0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                     0.01, 0.00, 0.01, 0.00, 0.01]
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, train_loss, 'b-', marker='o', markersize=4)
        ax.set_title('Training Loss Decrease')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss (MSE)')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        st.pyplot(fig)
        
        st.write("""
        The chart above shows how the model's error decreased during training. After about 10 epochs,
        the error stabilized at a very low level, indicating the model had learned the patterns in the data.
        """)
        
        # Evaluating performance exactly as in MLP.ipynb
        st.subheader("Model Performance")
        st.code("""
# คำนวณ Accuracy
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy

# แสดงผลเป็นเปอร์เซ็นต์
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Error Rate: {error_rate * 100:.2f}%")
        """, language="python")
        
        # Show actual output from the notebook
        st.code("""
Accuracy: 95.52%
Error Rate: 4.48%
""")
        
        st.markdown("""
        The model achieved high accuracy, correctly predicting rainfall 95.52% of the time.
        """)
        
        # Saving the model - exactly as in MLP.ipynb
        st.subheader("Saving the Model")
        st.code("""
with open('modelMLP.pkl', 'wb') as f:
    pickle.dump(model, f)
""", language="python")

        st.code("""
with open("scalerMLP.pkl", "wb") as file:
    pickle.dump(scaler, file)
""", language="python")

        st.code("""
with open("X_trainMLP.pkl", "wb") as file:
    X_train = pickle.dump(X_train,file)
with open("y_trainMLP.pkl", "wb") as file:
    y_train = pickle.dump(y_train,file)
""", language="python")

        st.write("""
        I saved the model, scaler, and training data for future use.
        """)

if __name__ == "__main__":
    main()