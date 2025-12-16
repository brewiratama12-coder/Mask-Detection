import streamlit as st
import os
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Deteksi Masker YOLOv8",
    page_icon="😷",
    layout="wide"
)

st.title("😷 Deteksi Masker Menggunakan YOLOv8")
st.markdown(
    "Aplikasi untuk mendeteksi **penggunaan masker** pada wajah "
    "menggunakan model **YOLOv8**."
)

st.sidebar.header("Pengaturan Model")
confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.0, 1.0, 0.5, 0.05
)

IS_CLOUD = os.getenv("STREAMLIT_CLOUD") is not None

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

MODEL_PATH = "models/best.pt"

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Model tidak ditemukan di '{MODEL_PATH}'.\n\nError: {e}")
    st.stop()

if IS_CLOUD:
    app_mode = "Upload Gambar"
    st.sidebar.info("Mode webcam dinonaktifkan di Streamlit Cloud.")
else:
    app_mode = st.sidebar.selectbox(
        "Pilih Mode Aplikasi",
        ["Webcam Live", "Upload Gambar"]
    )

if app_mode == "Webcam Live":
    st.subheader("📷 Deteksi Masker via Webcam")

    run = st.checkbox("Buka Kamera")

    frame_window = st.image([])

    if run:
        camera = cv2.VideoCapture(0)

        while run:
            ret, frame = camera.read()
            if not ret:
                st.warning("Kamera tidak dapat diakses.")
                break

            results = model(frame, conf=confidence, verbose=False)
            res_plotted = results[0].plot()

            rgb_frame = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            frame_window.image(rgb_frame)

        camera.release()
        st.info("Kamera dimatikan.")

elif app_mode == "Upload Gambar":
    st.subheader("🖼️ Upload Gambar untuk Deteksi Masker")

    uploaded_file = st.file_uploader(
        "Pilih file gambar",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Gambar Asli",
                use_container_width=True
            )

        if st.button("🔍 Deteksi Masker"):
            results = model(img_array, conf=confidence)
            res_plotted = results[0].plot()

            with col2:
                st.image(
                    res_plotted,
                    caption="Hasil Deteksi",
                    use_container_width=True
                )

            st.markdown("### 📊 Detail Deteksi")
            if len(results[0].boxes) == 0:
                st.info("Tidak ada objek terdeteksi.")
            else:
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    conf_score = float(box.conf[0])
                    label = results[0].names[cls]
                    st.write(f"- **{label}** : {conf_score:.2f}")
