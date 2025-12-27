# app.py - PhysioCheck: AI-Powered Physiotherapy Assistant

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
import time
import gc
from collections import deque
import base64
from PIL import Image
import io

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="PhysioCheck - AI Physiotherapy",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
.stApp { background: #0a0e27; }
[data-testid="stSidebar"] { background: #111827; }
h1, h2, h3 { color: #3b82f6 !important; font-weight: 600; }
.exercise-card {
    background: #1f2937; padding: 20px; border-radius: 12px;
    border: 1px solid #374151; margin: 10px 0;
}
.stButton>button {
    background: #3b82f6; color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ==================== MEDIAPIPE ====================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ==================== DIRECTORIES ====================
DATA_DIR = Path("exercise_data")
DATA_DIR.mkdir(exist_ok=True)
STEPS_DIR = DATA_DIR / "steps"
STEPS_DIR.mkdir(exist_ok=True)

# ==================== ANALYZER ====================
class StepBasedExerciseAnalyzer:
    def __init__(self, model_complexity=1):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.last_time = time.time()
        self.fps_q = deque(maxlen=20)

    def get_fps(self):
        now = time.time()
        fps = 1 / max(now - self.last_time, 1e-6)
        self.last_time = now
        self.fps_q.append(fps)
        return int(sum(self.fps_q) / len(self.fps_q))

    @staticmethod
    def angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
                  np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = abs(np.degrees(radians))
        return 360 - angle if angle > 180 else angle

    def extract_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        if not results.pose_landmarks:
            return None, None
        landmarks = [[l.x, l.y, l.z, l.visibility]
                     for l in results.pose_landmarks.landmark]
        return np.array(landmarks), results.pose_landmarks

    def calculate_angles(self, lm):
        return {
            "left_elbow": self.angle(lm[11][:2], lm[13][:2], lm[15][:2]),
            "right_elbow": self.angle(lm[12][:2], lm[14][:2], lm[16][:2]),
        }

# ==================== REP COUNTER ====================
class StepBasedRepCounter:
    def __init__(self):
        self.reps = 0
        self.state = 0

    def update(self, angles):
        if angles["left_elbow"] < 60:
            self.state = 1
        if angles["left_elbow"] > 150 and self.state == 1:
            self.reps += 1
            self.state = 0
        return self.reps

# ==================== SESSION STATE ====================
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## 🏃 PhysioCheck")
    st.markdown("AI Physiotherapy Assistant")
    st.markdown("---")

# ==================== MAIN ====================
st.title("PhysioCheck")
st.markdown("Live AI-based exercise analysis using MediaPipe Pose")

start, stop = st.columns(2)
with start:
    if st.button("▶ Start Camera"):
        st.session_state.camera_running = True
with stop:
    if st.button("⏹ Stop Camera"):
        st.session_state.camera_running = False

frame_box = st.empty()
rep_box = st.metric("Reps", 0)

# ==================== CAMERA LOOP ====================
if st.session_state.camera_running:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Cannot access camera")
        st.session_state.camera_running = False
    else:
        analyzer = StepBasedExerciseAnalyzer()
        counter = StepBasedRepCounter()

        while st.session_state.camera_running:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            lm, pose = analyzer.extract_landmarks(frame)

            if lm is not None:
                angles = analyzer.calculate_angles(lm)
                reps = counter.update(angles)
                rep_box.metric("Reps", reps)

                mp_drawing.draw_landmarks(
                    frame, pose, mp_pose.POSE_CONNECTIONS
                )

                cv2.putText(
                    frame, f"FPS: {analyzer.get_fps()}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (59, 130, 246), 2
                )

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_box.image(frame_rgb, channels="RGB", use_column_width=True)
            time.sleep(0.01)

        cap.release()
        gc.collect()

st.markdown("---")
st.caption("PhysioCheck • Streamlit Community Cloud • Live Webcam Enabled")

