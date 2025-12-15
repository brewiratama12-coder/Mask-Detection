import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Deteksi Masker YOLOv8",
    page_icon="😷",
    layout="wide"
)

st.title("😷 Live Mask Detection App")
st.markdown("Aplikasi deteksi penggunaan masker menggunakan **YOLOv8**.")

st.sidebar.header("Pengaturan Model")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)


@st.cache_resource
def load_model(path):
    return YOLO(path)

try:
    model = load_model("models/best.pt")
except Exception as e:
    st.error(f"Model tidak ditemukan di 'weights/best.pt'. Error: {e}")
    st.stop()

app_mode = st.sidebar.selectbox("Pilih Mode Aplikasi", ["Webcam Live", "Upload Gambar"])

if app_mode == "Webcam Live":
    st.subheader("Kamera Langsung")
    
    run = st.checkbox('Buka Kamera')
    frame_window = st.image([]) 

    camera = cv2.VideoCapture(0)
    
    while run:
        ret, frame = camera.read()
        if not ret:
            st.warning("Gagal membaca kamera/kamera tidak terdeteksi.")
            break

        results = model(frame, conf=confidence, verbose=False)

        res_plotted = results[0].plot()

        rgb_frame = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

        frame_window.image(rgb_frame)

    else:
        camera.release()
        st.write("Kamera berhenti.")

elif app_mode == "Upload Gambar":
    st.subheader("Upload Gambar untuk Deteksi")
    
    uploaded_file = st.file_uploader("Pilih file gambar...", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Gambar Asli", use_column_width=True)
            
        if st.button("Deteksi Masker"):
            img_array = np.array(image)

            results = model(img_array, conf=confidence)

            res_plotted = results[0].plot()

            with col2:
                st.image(res_plotted, caption="Hasil Deteksi", use_column_width=True)

            st.write("### Detail Deteksi:")
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = results[0].names[cls]
                st.write(f"- **{name}**: {conf:.2f}")