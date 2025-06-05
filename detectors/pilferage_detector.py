# detectors/pilferage_detector.py
import cv2
import numpy as np
import mediapipe as mp
import math
import time
from typing import List, Tuple, Dict, Any, Optional

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

class PilferageDetector:
    def __init__(self, model_complexity: int = 1, alert_cooldown_seconds: int = 3):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Behavior tracking
        self.suspicious_behaviors: List[str] = []
        self.person_timer: Dict[str, Dict[str, Any]] = {}
        self.concealment_timer: int = 0
        self.last_alert_time: float = 0.0
        self.alert_cooldown: int = alert_cooldown_seconds
        self.current_alerts: List[str] = []

    def calculate_distance(self, point1: Any, point2: Any) -> float:
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    def detect_concealment_gesture(self, pose_landmarks: Any, hand_landmarks_list: List[Any]) -> bool:
        if not pose_landmarks or not hand_landmarks_list:
            return False

        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]

        torso_center_x = (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4
        torso_center_y = (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4
        torso_ref_point = Point(x=torso_center_x, y=torso_center_y)

        for hand_landmarks in hand_landmarks_list:
            for landmark in hand_landmarks.landmark:
                distance_to_torso = self.calculate_distance(landmark, torso_ref_point)
                if distance_to_torso < 0.2:
                    return True
        return False

    def detect_loitering(self, pose_landmarks: Any, person_id: str = "person_1") -> bool:
        if not pose_landmarks:
            return False

        center_x = (pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x +
                    pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2
        center_y = (pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y +
                    pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
        current_time = time.time()

        if person_id not in self.person_timer:
            self.person_timer[person_id] = {
                'start_time': current_time,
                'position': (center_x, center_y),
                'movement_threshold': 0.05
            }
            return False

        prev_pos = self.person_timer[person_id]['position']
        distance_moved = math.sqrt((center_x - prev_pos[0])**2 + (center_y - prev_pos[1])**2)

        if distance_moved > self.person_timer[person_id]['movement_threshold']:
            self.person_timer[person_id]['start_time'] = current_time
            self.person_timer[person_id]['position'] = (center_x, center_y)
            return False

        stationary_time = current_time - self.person_timer[person_id]['start_time']
        return stationary_time > 5

    def detect_suspicious_pose(self, pose_landmarks: Any) -> bool:
        if not pose_landmarks:
            return False

        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]

        shoulder_midpoint_y = (left_shoulder.y + right_shoulder.y) / 2
        lean_factor = abs(nose.y - shoulder_midpoint_y)

        return lean_factor > 0.15

    def draw_alerts(self, image: np.ndarray, alerts: List[str]):
        y_offset = 30
        for alert_message in alerts:
            cv2.putText(image, f"ALERT: {alert_message}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            y_offset += 30

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        self.current_alerts = []
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False

        pose_results = self.pose.process(rgb_image)
        hand_results = self.hands.process(rgb_image)

        rgb_image.flags.writeable = True

        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

        if pose_results.pose_landmarks:
            if self.detect_loitering(pose_results.pose_landmarks):
                self.current_alerts.append("Loitering")
            if self.detect_suspicious_pose(pose_results.pose_landmarks):
                self.current_alerts.append("Suspicious Posture")
            if hand_results.multi_hand_landmarks:
                if self.detect_concealment_gesture(pose_results.pose_landmarks, hand_results.multi_hand_landmarks):
                    self.concealment_timer += 1
                    if self.concealment_timer > 5:
                        self.current_alerts.append("Concealment Gesture")
                else:
                    self.concealment_timer = 0
            else:
                self.concealment_timer = 0
        else:
            self.person_timer.clear()
            self.concealment_timer = 0

        current_time = time.time()
        if self.current_alerts and (current_time - self.last_alert_time) > self.alert_cooldown:
            self.draw_alerts(frame, self.current_alerts)
            self.last_alert_time = current_time

        return frame, self.current_alerts