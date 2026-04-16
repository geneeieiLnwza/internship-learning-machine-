import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page config
st.set_page_config(page_title="FarmTwin Simulator", page_icon="🌱", layout="wide")

# Custom CSS for aesthetics
st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    h1 {
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Load model and encoder
@st.cache_resource
def load_models():
    model = joblib.load('models/farmtwin_model.pkl')
    encoder = joblib.load('models/farmtwin_encoder.pkl')
    return model, encoder

try:
    model, encoder = load_models()
except FileNotFoundError:
    st.error("ไม่พบไฟล์โมเดล กรุณารันคำสั่ง `python3 scripts/train_model.py` ก่อนครับ")
    st.stop()

st.title("🌱 FarmTwin: Digital Twin Agriculture Simulator")
st.markdown("ระบบจำลองสถานการณ์ (What-If Analysis) เพื่อทำนายผลผลิตการเกษตร โดยให้คุณสามารถทดลองปรับปัจจัยที่ควบคุมได้ (น้ำและปุ๋ย) เพื่อดูแนวโน้มผลผลิตที่เปลี่ยนไป")

# Sidebar for base Environment properties
st.sidebar.header("🌍 1. สภาพแวดล้อม (Environment)")
crop = st.sidebar.selectbox("เลือกชนิดพืช (Crop Type)", ['Rice', 'Wheat', 'Maize', 'Soybean'])
temp = st.sidebar.slider("อุณหภูมิ (°C)", 10.0, 45.0, 25.0, step=0.5)
rainfall = st.sidebar.slider("ปริมาณฝน (mm/season)", 0.0, 2000.0, 800.0, step=10.0)

# Main area for Controllables (What-if)
st.header("🎛️ 2. ปัจจัยควบคุม (Farm Management)")
col1, col2 = st.columns(2)

with col1:
    st.subheader("💧 ระบบชลประทาน")
    irrigation = st.slider("การให้น้ำเพิ่ม (Irrigation mm)", 0.0, 1000.0, 300.0, step=10.0)

with col2:
    st.subheader("🧪 การให้ปุ๋ย (Fertilizers)")
    n_fert = st.slider("Nitrogen (N) kg/ha", 0.0, 300.0, 120.0, step=5.0)
    p_fert = st.slider("Phosphorous (P) kg/ha", 0.0, 150.0, 40.0, step=5.0)
    k_fert = st.slider("Potassium (K) kg/ha", 0.0, 150.0, 40.0, step=5.0)

# Prediction Logic
def predict_yield(c, t, r, i, n, p, k):
    # Encode crop
    encoded_c = encoder.transform(pd.DataFrame([[c]], columns=['Crop_Type']))
    c_df = pd.DataFrame(encoded_c, columns=encoder.get_feature_names_out(['Crop_Type']))
    
    # Numeric features
    num_df = pd.DataFrame({'Temperature_C': [t], 'Rainfall_mm': [r], 'Irrigation_mm': [i], 
                           'N_Fertilizer': [n], 'P_Fertilizer': [p], 'K_Fertilizer': [k]})
    
    # Combine
    X_input = pd.concat([num_df, c_df], axis=1)
    prediction = model.predict(X_input)[0]
    return max(0, prediction)

# Calculate prediction
current_yield = predict_yield(crop, temp, rainfall, irrigation, n_fert, p_fert, k_fert)

st.divider()
st.header("📊 3. ผลลัพธ์การจำลอง (Simulation Outcome)")

# Display Metric
st.metric(label=f"ปริมาณผลผลิตที่คาดการณ์ของ {crop} (Predicted Yield)", value=f"{current_yield:,.2f} kg/ha")

st.info("💡 **คำแนะนำในฐานะ Digital Twin:** ลองค่อยๆ ลดหรือเพิ่มแถบเลื่อนปริมาณน้ำ (Irrigation) หรือปุ๋ย (Nitrogen) ดูครับ จะสังเกตเห็นว่าถ้าใส่น้อยไปหรือมากเกินไป ผลผลิตก็อาจจะตกได้ ซึ่งเป็นกลไกที่โมเดล AI เรียนรู้จากข้อมูลจำลองที่เราสร้างไว้")
