# app.py — PhysioCheck (Streamlit Cloud compatible)

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from PIL import Image

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="PhysioCheck",
    page_icon="🏃‍♂️",
    layout="wide"
)

# ==================== STYLES ====================
st.markdown("""
<style>
.stApp { background: #0a0e27; }
h1, h2, h3 { color: #3b82f6; }
</style>
""", unsafe_allow_html=True)

# ==================== MEDIAPIPE ====================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ==================== ANALYZER ====================
class PoseAnalyzer:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.last_time = time.time()
        self.fps_q = deque(maxlen=10)

    def get_fps(self):
        now = time.time()
        fps = 1 / max(now - self.last_time, 1e-6)
        self.last_time = now
        self.fps_q.append(fps)
        return int(sum(self.fps_q) / len(self.fps_q))

    @staticmethod
    def angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])
        deg = abs(np.degrees(rad))
        return 360 - deg if deg > 180 else deg

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)

        if not res.pose_landmarks:
            return frame_bgr, None

        lm = res.pose_landmarks.landmark

        # Simple elbow angle example
        left_elbow = self.angle(
            [lm[11].x, lm[11].y],
            [lm[13].x, lm[13].y],
            [lm[15].x, lm[15].y],
        )

        mp_drawing.draw_landmarks(
            frame_bgr,
            res.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        return frame_bgr, left_elbow

# ==================== REP COUNTER ====================
class RepCounter:
    def __init__(self):
        self.state = 0
        self.reps = 0

    def update(self, angle):
        if angle is None:
            return self.reps

        if angle < 60:
            self.state = 1
        elif angle > 150 and self.state == 1:
            self.reps += 1
            self.state = 0

        return self.reps

# ==================== APP ====================
st.title("PhysioCheck")
st.markdown("AI Physiotherapy Assistant (Streamlit Cloud Demo)")

st.info(
    "This demo uses **browser camera snapshots** (`st.camera_input`). "
    "Live OpenCV webcam streams are not supported on Streamlit Cloud."
)

analyzer = PoseAnalyzer()
counter = RepCounter()

frame_file = st.camera_input("📸 Capture frame")

if frame_file is not None:
    image = Image.open(frame_file).convert("RGB")
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    processed, elbow_angle = analyzer.process(frame)
    reps = counter.update(elbow_angle)

    col1, col2 = st.columns(2)
    with col1:
        st.image(
            cv2.cvtColor(processed, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_column_width=True
        )
    with col2:
        st.metric("Elbow Angle", f"{int(elbow_angle) if elbow_angle else '--'}°")
        st.metric("Reps", reps)
        st.metric("FPS", analyzer.get_fps())

st.markdown("---")
st.caption("PhysioCheck • Streamlit Community Cloud • Headless OpenCV")

