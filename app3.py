import streamlit as st
import cv2
import easyocr
import pandas as pd
import numpy as np
import os
from datetime import datetime
import plotly.express as px

# --- CORE LOGIC (Kept exactly as requested) ---
class AQIAutoDetector:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.bps = {
            "PM25": [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200), (91, 120, 201, 300), (121, 250, 301, 400), (250, 500, 401, 500)],
            "PM10": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200), (251, 350, 201, 300), (351, 430, 301, 400), (430, 500, 401, 500)],
            "CO":   [(0, 1, 0, 50), (1.1, 2, 51, 100), (2.1, 10, 101, 200), (10.1, 17, 201, 300), (17.1, 34, 301, 400), (34.1, 100, 401, 500)]
        }

    def get_sub_index(self, conc, pollutant):
        if pollutant not in self.bps or conc <= 0: return 0
        for (clo, chi, ilo, ihi) in self.bps[pollutant]:
            if clo <= conc <= chi:
                return ((ihi - ilo) / (chi - clo)) * (conc - clo) + ilo
        return 500

    def process_image_data(self, img, filename):
        raw_results = self.reader.readtext(img, allowlist='0123456789.', detail=1)
        h, w = img.shape[:2]
        detected_numbers = []
        for (bbox, text, prob) in raw_results:
            if text.replace('.', '', 1).isdigit():
                center_y = sum([p[1] for p in bbox]) / 4
                center_x = sum([p[0] for p in bbox]) / 4
                if 0.35 * w < center_x < 0.65 * w:
                    detected_numbers.append((center_y, float(text)))
        
        detected_numbers.sort()
        row = {"Filename": filename, "Timestamp": datetime.now().isoformat()}
        row.update({"PM25": 0, "PM10": 0, "CO": 0, "CO2": 0})
        
        if len(detected_numbers) >= 4:
            target_vals = detected_numbers[-4:]
            row["PM25"], row["PM10"], row["CO"], row["CO2"] = [v[1] for v in target_vals]

        si_pm25 = self.get_sub_index(row["PM25"], "PM25")
        si_pm10 = self.get_sub_index(row["PM10"], "PM10")
        si_co   = self.get_sub_index(row["CO"], "CO")
        row["Final_AQI"] = int(round(max(si_pm25, si_pm10, si_co)))
        return row

# --- NEW: AQI Status Function ---
def get_aqi_status(aqi):
    """Categorize AQI and return status string and associated color"""
    if aqi <= 50:
        return "Good", "#00e400"
    elif aqi <= 100:
        return "Satisfactory", "#92d050"
    elif aqi <= 200:
        return "Moderate", "#ffff00"
    elif aqi <= 300:
        return "Poor", "#ff9900"
    elif aqi <= 400:
        return "Very Poor", "#ff0000"
    else:
        return "Severe", "#7e0023"

# --- STREAMLIT UI ---
st.set_page_config(page_title="AQI Vision", layout="wide")

# Initialize Pipeline
@st.cache_resource
def load_ocr():
    return AQIAutoDetector()

detector = load_ocr()

# Sidebar: Image Upload
st.sidebar.title("📤 Image Input")
uploaded_file = st.sidebar.file_uploader("Upload Air Monitor Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.sidebar.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.sidebar.button("Process Image"):
        new_data = detector.process_image_data(image, uploaded_file.name)
        csv_path = "aqi_logs.csv"
        df_new = pd.DataFrame([new_data])
        df_new.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
        st.sidebar.success("✅ Processed and Logged!")

# Main Dashboard
st.title("🌱 Real-Time Air Quality Dashboard")

try:
    df = pd.read_csv("aqi_logs.csv")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    latest = df.iloc[-1]
    
    # --- NEW: UI Status Header ---
    status_text, status_color = get_aqi_status(latest['Final_AQI'])
    
    st.markdown(f"""
        <div style="background-color: {status_color}; padding: 20px; border-radius: 10px; text-align: center;">
            <h1 style="color: black; margin: 0;">Status: {status_text}</h1>
            <h3 style="color: black; margin: 0;">AQI: {int(latest['Final_AQI'])}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()

    # Row 1: Metrics
    st.subheader("📍 Latest Extraction Results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PM 2.5", f"{latest['PM25']} µg/m³")
    m2.metric("PM 10", f"{latest['PM10']} µg/m³")
    m3.metric("CO", f"{latest['CO']} ppm")
    m4.metric("CO2", f"{latest['CO2']} ppm")

    st.divider()

    # Row 2: Graph
    st.subheader("📈 Pollutant Trends")
    fig = px.line(df, x="Timestamp", y=["Final_AQI", "PM25", "PM10"], 
                  labels={"value": "Concentration/Index", "variable": "Metric"},
                  title="Historical AQI Data Trend")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Row 3: Table
    st.subheader("📜 Detailed Logs")
    st.dataframe(df.sort_values(by="Timestamp", ascending=False), use_container_width=True)

except Exception as e:
    st.info("Upload an image on the left and click 'Process' to start generating data.")