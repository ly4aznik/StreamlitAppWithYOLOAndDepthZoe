import streamlit as st
import cv2
import tempfile
import time
from PIL import Image
from depth_yolo import DepthYOLO
import numpy as np


st.set_page_config(page_title="Детектор с использованием карты глубины", layout="wide")
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-size: 20px !important;
    }
    .stSlider > div > div {
        font-size: 20px !important;
    }
    .stCheckbox > label {
        font-size: 20px !important;
    }
    .stTitle {
        font-size: 32px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Детекция объектов с глубиной")

# Инициализация состояния
if "confidence_threshold" not in st.session_state:
    st.session_state.confidence_threshold = 0.3
if "yolo" not in st.session_state:
    st.session_state.yolo = DepthYOLO("yolo_weights/yolov5m_best.pt")

# Ползунки на одном уровне
col1, col2 = st.columns(2)

with col1:
    old_conf = st.session_state.confidence_threshold
    st.session_state.confidence_threshold = st.slider(
        "Порог уверенности", 0.0, 1.0, old_conf, 0.01
    )
    if old_conf != st.session_state.confidence_threshold:
        st.session_state.yolo.change_confidence(st.session_state.confidence_threshold)

with col2:
    scale = st.slider("Масштаб отображения", 0.1, 2.0, 1.0, 0.1)

# Выбор режима отображения
show_depth = st.checkbox("Показать карту глубины", value=True)

# Загрузка файла — изображение или видео
uploaded_file = st.file_uploader(
    "Загрузите изображение или видео для обработки",
    type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"]
)

stream_url = st.text_input("URL видеопотока (RTSP/HTTP):")
start_stream = st.button("Запустить поток")
stop_stream = st.button("Остановить поток")

if start_stream and stream_url:
    cap = cv2.VideoCapture(stream_url)
    st.session_state.streaming = True

if stop_stream:
    st.session_state.streaming = False

if st.session_state.get("streaming", False):
    cap = cap if 'cap' in locals() else cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)
    placeholder = st.empty()
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Не удалось получить кадр из потока.")
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if show_depth:
            rgb_box, depth_box = st.session_state.yolo.process_image_with_depth_map(frame_rgb)
        else:
            rgb_box = st.session_state.yolo.process_image(frame_rgb)
            depth_box = None

        h, w = rgb_box.shape[:2]
        rgb_out = cv2.resize(rgb_box, (int(w*scale), int(h*scale)))
        if show_depth and depth_box is not None:
            dh, dw = depth_box.shape[:2]
            depth_out = cv2.resize(depth_box, (int(dw*scale), int(dh*scale)))
            placeholder.image([rgb_out, depth_out], caption=["RGB","Depth"], use_column_width=False)
        else:
            placeholder.image(rgb_out, caption="RGB", use_column_width=False)

    cap.release()
    st.session_state.streaming = False

if uploaded_file:
    file_bytes = uploaded_file.read()
    if uploaded_file.type.startswith("image"):
        img_bgr = cv2.imdecode(
            np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR
        )
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if show_depth:
            rgb_box, depth_box = st.session_state.yolo.process_image_with_depth_map(img_rgb)
        else:
            rgb_box = st.session_state.yolo.process_image(img_rgb)
            depth_box = None

        h, w = rgb_box.shape[:2]
        new_size = (int(w*scale), int(h*scale))
        rgb_box_resized = cv2.resize(rgb_box, new_size)
        st.image(rgb_box_resized, caption="Результат на RGB", use_column_width=False)

        if show_depth and depth_box is not None:
            db_h, db_w = depth_box.shape[:2]
            depth_resized = cv2.resize(depth_box, (int(db_w*scale), int(db_h*scale)))
            st.image(depth_resized, caption="Результат на карте глубины", use_column_width=False)

    else:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file_bytes)
        cap = cv2.VideoCapture(tfile.name)

        if "playing" not in st.session_state or st.session_state.last_filename != uploaded_file.name:
            st.session_state.playing = False
            st.session_state.frame_idx = 0
            st.session_state.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            st.session_state.cache = {}
            st.session_state.last_filename = uploaded_file.name
        cap.release()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Пуск"):
                st.session_state.playing = True
        with col2:
            if st.button("⏸️ Пауза"):
                st.session_state.playing = False

        if not st.session_state.playing:
            st.session_state.frame_idx = st.slider(
                "Кадр", 0, st.session_state.total_frames - 1, st.session_state.frame_idx
            )

        cap = cv2.VideoCapture(tfile.name)
        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            st.error("Не удалось прочитать кадр.")
            st.session_state.playing = False
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if st.session_state.frame_idx not in st.session_state.cache:
                if show_depth:
                    rgb_box, depth_box = st.session_state.yolo.process_image_with_depth_map(frame_rgb)
                    st.session_state.cache[st.session_state.frame_idx] = (rgb_box, depth_box)
                else:
                    rgb_box = st.session_state.yolo.process_image(frame_rgb)
                    st.session_state.cache[st.session_state.frame_idx] = (rgb_box, None)

            rgb_box, depth_box = st.session_state.cache[st.session_state.frame_idx]

            h, w = rgb_box.shape[:2]
            rgb_resized = cv2.resize(rgb_box, (int(w*scale), int(h*scale)))
            st.image(rgb_resized, caption=f"Кадр {st.session_state.frame_idx}", use_column_width=False)

            if show_depth and depth_box is not None:
                dh, dw = depth_box.shape[:2]
                depth_resized = cv2.resize(depth_box, (int(dw*scale), int(dh*scale)))
                st.image(depth_resized, caption=f"Карта глубины {st.session_state.frame_idx}", use_column_width=False)

            if st.session_state.playing:
                st.session_state.frame_idx += 1
                if st.session_state.frame_idx >= st.session_state.total_frames:
                    st.success("Воспроизведение завершено.")
                    st.session_state.playing = False
                else:
                    st.rerun()
