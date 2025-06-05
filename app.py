# app.py
import streamlit as st
import os
import cv2
import time
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image
import warnings
import pandas as pd

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Import detector and utilities
from detectors.pilferage_detector import PilferageDetector
from utils.file_utils import log_detection

# Helper function to convert frame number to video timestamp
def frame_to_timestamp(frame_num, fps):
    total_seconds = frame_num / fps
    return str(timedelta(seconds=int(total_seconds)))

# Page configuration
st.set_page_config(page_title="Vision Sense - Pilferage Detection", layout="wide")
st.markdown("<h1 style='text-align: center;'>👁️ Vision Sense - Pilferage Detection</h1>", unsafe_allow_html=True)

# Paths
INPUT_DIR = "inputs"
PROCESSED_DIR = "outputs/processed"
ALERT_DIR = "outputs/alerts"
LOG_DIR = "outputs/logs"

for d in [INPUT_DIR, PROCESSED_DIR, ALERT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

def clear_directory(directory):
    for f in os.listdir(directory):
        file_path = os.path.join(directory, f)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            st.warning(f"Could not delete {file_path}: {e}")

def save_uploaded_video(uploaded_file):
    video_path = os.path.join(INPUT_DIR, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return video_path

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else 1000
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(PROCESSED_DIR, "processed_" + os.path.basename(video_path))
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    detector = PilferageDetector(model_complexity=1, alert_cooldown_seconds=3)

    frame_count = 0
    alert_segments = []
    current_alert_segment = None

    # Initialize progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, alerts = detector.process_frame(frame)

        # Log alerts
        if alerts:
            timestamp = frame_to_timestamp(frame_count, fps)
            for alert in alerts:
                log_detection(alert, timestamp)
            if current_alert_segment is None:
                current_alert_segment = {
                    'start_frame': max(0, frame_count),
                    'end_frame': frame_count,
                    'alerts': set(alerts)
                }
            else:
                current_alert_segment['end_frame'] = frame_count
                current_alert_segment['alerts'].update(alerts)
        else:
            if current_alert_segment is not None:
                current_alert_segment['end_frame'] = frame_count
                alert_segments.append(current_alert_segment)
                current_alert_segment = None

        out.write(processed_frame)
        frame_count += 1

        # Update progress
        progress = int((frame_count / total_frames) * 100)
        progress_bar.progress(min(progress, 100))
        status_text.text(f"Processing Frame {frame_count} / {total_frames}")

    # Save any remaining alert segment
    if current_alert_segment:
        alert_segments.append(current_alert_segment)

    cap.release()
    out.release()
    progress_bar.empty()
    status_text.empty()

    # Save alert clips
    input_path = Path(video_path)
    for i, segment in enumerate(alert_segments):
        alert_types = "_".join(sorted(segment['alerts']))
        clip_filename = f"{input_path.stem}_alert_{i+1}_{alert_types}.mp4"
        clip_path = os.path.join(ALERT_DIR, clip_filename)
        extract_alert_clip(video_path, segment['start_frame'], segment['end_frame'], clip_path, fps)

    return out_path, alert_segments

def extract_alert_clip(video_source, start_frame, end_frame, output_path, fps):
    cap = cv2.VideoCapture(video_source)
    frame_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(end_frame - start_frame + 1):
        ret, frame = cap.read()
        if ret:
            writer.write(frame)
    writer.release()
    cap.release()

def display_video(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        stframe.image(img, caption="Processed Video", use_container_width=True)
        time.sleep(0.03)
    cap.release()

# def display_alert_clips(alert_dir):
#     alert_files = [f for f in os.listdir(alert_dir) if f.endswith(".mp4")]
#     if not alert_files:
#         st.info("No alert clips detected.")
#         return
#     st.subheader("🚨 Alert Clips")
#     for clip in alert_files:
#         st.video(os.path.join(alert_dir, clip))

def display_logs(log_dir):
    log_path = os.path.join(log_dir, "detections.csv")
    if not os.path.exists(log_path):
        st.warning("No detection logs found.")
        return
    df = pd.read_csv(log_path)
    st.subheader("📊 Detection Logs")
    st.dataframe(df[["timestamp", "event_type"]], use_container_width=True)

def main():
    st.markdown("### Upload Video for Analysis")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        clear_directory(INPUT_DIR)
        clear_directory(PROCESSED_DIR)
        clear_directory(ALERT_DIR)

        video_path = save_uploaded_video(uploaded_file)
        st.success(f"Uploaded: {uploaded_file.name}")

        if st.button("Process Video"):
            with st.spinner("Processing video... This may take a few moments."):
                processed_path, alert_segments = process_video(video_path)
                st.success("✅ Processing complete!")

            col1, col2 = st.columns([3, 2])  # 3: video, 2: logs

            with col1:
                display_video(processed_path)

            with col2:
                display_logs(LOG_DIR)

            #display_alert_clips(ALERT_DIR)

if __name__ == "__main__":
    main()