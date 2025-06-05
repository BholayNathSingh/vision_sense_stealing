# utils/file_utils.py
import os
import csv
from datetime import datetime

LOG_DIR = "outputs/logs"
ALERT_DIR = "outputs/alerts"
PROCESSED_DIR = "outputs/processed"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ALERT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def log_detection(event_type: str, timestamp: str):
    log_path = os.path.join(LOG_DIR, "detections.csv")
    file_exists = os.path.isfile(log_path)

    with open(log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "event_type"])
        writer.writerow([timestamp, event_type])