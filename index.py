import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import streamlit.components.v1 as components

# Set up your page config first
st.set_page_config(page_title="ML Project", page_icon="🔍", layout="wide")

# สร้าง container หลัก
main_container = st.container()

with main_container:
    # หัวข้อหลักพร้อมจัดกลาง
    st.markdown("<h1 style='text-align: center;'>🔍 Overview และเครดิตโปรเจค</h1>", unsafe_allow_html=True)
    
    # เพิ่มเส้นคั่นและระยะห่าง
    st.markdown("<hr style='margin-bottom: 30px;'>", unsafe_allow_html=True)

    # สร้างคอลัมน์สองคอลัมน์
    col1, gap, col2 = st.columns([10, 1, 8])

    with col1:
        st.markdown("<h2 style='text-align: center; margin-bottom: 20px;'>เกี่ยวกับโปรเจค</h2>", unsafe_allow_html=True)
        st.markdown("""
        โปรเจคนี้เป็นการวิเคราะห์และสร้างโมเดล Machine Learning โดยใช้ชุดข้อมูลจาก Kaggle 
        เพื่อทำนายราคาบ้านและวิเคราะห์สภาพอากาศ เราได้จำลองสถานการณ์การทำงานกับข้อมูลในโลกจริง
        โดยการเพิ่ม missing values และ wrong values เข้าไปในชุดข้อมูล
        """)
        
        st.markdown("<h3 style='margin-top: 20px;'>แหล่งข้อมูลที่ใช้:</h3>", unsafe_allow_html=True)
        st.markdown("""
        - [Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)
        - [Weather Dataset](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)
        """)
        
        st.markdown("<h3 style='margin-top: 20px;'>แนวทางการพัฒนา</h3>", unsafe_allow_html=True)
        st.markdown("""
        1. **การเตรียมข้อมูล**: ChatGPT ช่วยในการสร้าง missing values และ wrong values เพื่อtrain model
        2. **การสร้างโมเดล**: พัฒนาโมเดล ML และ Neural Network เพื่อทำนายราคาบ้านและวิเคราะห์สภาพอากาศ
        3. **การตกแต่งและอธิบาย**: ใช้ Claude AI ช่วยในการออกแบบ UI และอธิบายโมเดล Neural Network
        4. **Demo model**: จัดทำ demo โดย ใช้ claude AI ช่วย guide บางส่วน
        """)

    with col2:
        st.markdown("<h2 style='text-align: center; margin-bottom: 20px;'>เครดิต</h2>", unsafe_allow_html=True)
        
        st.markdown("<h3 style='margin-top: 20px;'>Credit:</h3>", unsafe_allow_html=True)
        st.markdown("""
        - **Claude AI**:
          - ช่วยในการออกแบบ UI
          - ช่วยGuideหน้า อธิบายโมเดล Neural Network
          - ช่วยคำนวณ Feature Importance
        
        - **ChatGPT**:
          - ช่วยในการสร้าง missing values ,wrong values และสร้างบาง column ใน house price เพื่อทดสอบโมเดล
          - ช่วย Guide Model Training
        """)
        
        st.markdown("<h3 style='margin-top: 20px;'>แหล่งข้อมูลและเอกสารอ้างอิง:</h3>", unsafe_allow_html=True)
        st.markdown("""
        - **Streamlit Documentation**:
          - ใช้เป็นแนวทางในการพัฒนา UI
          - อ้างอิงในการใช้ components ต่างๆ
        
        - **Streamlit Guide**:
          - แนวทางการออกแบบ Data Dashboard
          - เทคนิคการแสดงผลข้อมูลแบบ Interactive
        """)
        
        
    # เพิ่มเส้นคั่นด้านล่าง
    st.markdown("<hr style='margin: 40px 0 20px 0;'>", unsafe_allow_html=True)
    
    # ส่วนเอกสารอ้างอิง
    st.markdown("<h2 style='text-align: center; margin-bottom: 30px;'>เอกสารอ้างอิงและแหล่งเรียนรู้</h2>", unsafe_allow_html=True)
    
    # แบ่งลิงก์เป็น 2 คอลัมน์
    ref_col1, ref_col2 = st.columns(2)
    
    with ref_col1:
        st.markdown("<h3 style='text-align: center; margin-bottom: 15px;'>Streamlit & UI</h3>", unsafe_allow_html=True)
        st.markdown("""
        - [Streamlit Documentation](https://docs.streamlit.io/)
        - [Streamlit Components](https://streamlit.io/components)
        - [Streamlit Gallery](https://streamlit.io/gallery)
        """)
    
    with ref_col2:
        st.markdown("<h3 style='text-align: center; margin-bottom: 15px;'>Data & ML</h3>", unsafe_allow_html=True)
        st.markdown("""
        - [Kaggle Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)
        - [Kaggle Weather Dataset](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)
        - [Guide to Neural Networks for Regression](https://www.tensorflow.org/tutorials/keras/regression)
        - [Feature Importance in Machine Learning](https://scikit-learn.org/stable/modules/permutation_importance.html)
        """)
    
    # ส่วนท้ายเพจ
    st.markdown("<div style='text-align: center; margin-top: 40px;'>", unsafe_allow_html=True)
    st.markdown("<p><strong>โปรเจคนี้เป็นส่วนหนึ่งของการเรียนรู้ Machine Learning และการวิเคราะห์ข้อมูล</strong></p>", unsafe_allow_html=True)
    st.markdown("<p style='color: #666; font-size: 0.8em;'>© 2025 - สงวนลิขสิทธิ์</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)