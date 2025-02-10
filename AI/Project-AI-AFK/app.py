import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# โหลดโมเดล AI
model = joblib.load("model.pkl")  # บันทึกโมเดลหลัง Train ด้วย joblib

# โหลด LabelEncoder
try:
    label_encoder = joblib.load("label_encoder.pkl")  # โหลด LabelEncoder ที่ใช้ตอน Train
except FileNotFoundError:
    st.error(" ไม่พบไฟล์ 'label_encoder.pkl' กรุณาตรวจสอบว่าคุณได้บันทึก LabelEncoder และวางไฟล์ในโฟลเดอร์เดียวกับโค้ด")

# โหลดข้อมูลนักเตะ
try:
    df = pd.read_csv("EPL_20_21.csv")  # ไฟล์ข้อมูลของคุณ
except FileNotFoundError:
    st.error("⚠️ ไม่พบไฟล์ 'EPL_20_21.csv' กรุณาตรวจสอบว่าไฟล์อยู่ในโฟลเดอร์เดียวกับโค้ด")
    df = pd.DataFrame()  # กรณีไม่มีไฟล์ ให้ df เป็น DataFrame ว่าง

# ตั้งค่าหน้าตา UI
st.set_page_config(page_title="Football AI Prediction", layout="wide")

# Header
st.title("⚽AI Player Performance Prediction")
st.markdown("## วิเคราะห์ฟอร์มนักเตะ ด้วย Machine Learning")

# รับค่าจากผู้ใช้
st.sidebar.header("📊 กรอกข้อมูลนักเตะ")
age = st.sidebar.slider("Age", 18, 40, 25)
matches = st.sidebar.number_input("Matches", min_value=0, max_value=50, value=10)
starts = st.sidebar.number_input("Starts", min_value=0, max_value=50, value=5)
mins = st.sidebar.number_input("Minutes", min_value=0, max_value=4500, value=900)  # นาทีที่เล่น
goals = st.sidebar.number_input("Goals", min_value=0, max_value=50, value=5)
assists = st.sidebar.number_input("Assists", min_value=0, max_value=50, value=3)

#  ทำ Prediction
if st.sidebar.button("Predict Performance"):
    if 'label_encoder' in locals():  # ตรวจสอบว่า label_encoder ถูกโหลดสำเร็จหรือไม่
        # สร้าง input_data ที่ประกอบด้วย 6 features
        input_data = pd.DataFrame({
            'Age': [age],
            'Matches': [matches],
            'Starts': [starts],
            'Mins': [mins],
            'Goals': [goals],
            'Assists': [assists]
        })
        
        # ทำนายผล
        prediction = model.predict(input_data)
        
        # แปลงผลลัพธ์จากตัวเลขกลับเป็นคลาส (Good, Average, Poor)
        predicted_class = label_encoder.inverse_transform(prediction)[0]
        
        st.sidebar.success(f"🔮 Predicted Performance: {predicted_class}")
    else:
        st.sidebar.error("⚠️ ไม่สามารถทำการพยากรณ์ได้ เนื่องจาก LabelEncoder ไม่ถูกโหลด")

# 📊 แสดงกราฟข้อมูลนักเตะ
if not df.empty:
    st.subheader("📈 สถิตินักเตะในลีก")

    # 1️⃣ กราฟ Goals vs Assists
    fig1, ax1 = plt.subplots()
    sns.scatterplot(x=df["Goals"], y=df["Assists"], hue=df["Position"], palette="coolwarm", ax=ax1)
    ax1.set_xlabel("Goals")
    ax1.set_ylabel("Assists")
    ax1.set_title("Goals vs Assists")
    st.pyplot(fig1)

    # 2️⃣ กราฟ Age Distribution
    fig2, ax2 = plt.subplots()
    sns.histplot(df["Age"], bins=15, kde=True, color="blue", ax=ax2)
    ax2.set_xlabel("Age")
    ax2.set_ylabel("Count")
    ax2.set_title("Age Distribution of Players")
    st.pyplot(fig2)

#  Footer
st.markdown("---")
st.markdown("📌 **Developed by Panuthep & Tosapat | Football AI Project**")
