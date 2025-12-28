# app.py - PhysioCheck (FIXED for Streamlit Community Cloud)

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
import base64
from PIL import Image
import io
import av

from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="PhysioCheck - AI Physiotherapy",
    page_icon="🏃‍♂️",
    layout="wide",
)

# ==================== MEDIAPIPE ====================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ==================== DIRECTORIES ====================
DATA_DIR = Path("exercise_data")
DATA_DIR.mkdir(exist_ok=True)
STEPS_DIR = DATA_DIR / "steps"
STEPS_DIR.mkdir(exist_ok=True)

# ==================== RTC CONFIG ====================
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ==================== HELPERS ====================
def get_image_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

def display_image_frame(path, size="120px"):
    b64 = get_image_base64(path)
    if not b64:
        return
    st.markdown(f"""
    <div style="width:{size};height:{size};background:#111827;border:2px solid #374151;
                border-radius:10px;display:flex;align-items:center;justify-content:center;
                overflow:hidden;margin:6px auto;">
        <img src="data:image/jpeg;base64,{b64}" style="max-width:100%;max-height:100%;"/>
    </div>
    """, unsafe_allow_html=True)

# ==================== ANALYZER ====================
class ExerciseAnalyzer:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def get_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if res.pose_landmarks:
            lms = [[l.x, l.y, l.z, l.visibility] for l in res.pose_landmarks.landmark]
            return np.array(lms), res.pose_landmarks
        return None, None

    @staticmethod
    def angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        r = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        ang = abs(r * 180 / np.pi)
        return 360-ang if ang > 180 else ang

    def get_angles(self, l):
        return {
            "left_elbow": self.angle(l[11][:2], l[13][:2], l[15][:2]),
            "right_elbow": self.angle(l[12][:2], l[14][:2], l[16][:2]),
        }

    def compare(self, curr, ref, tol=20):
        c = t = 0
        for k, v in ref.items():
            if k in curr:
                t += 1
                if abs(curr[k] - v) <= tol:
                    c += 1
        return (c / t * 100) if t else 0

# ==================== VIDEO PROCESSOR ====================
class VideoProcessor:
    def __init__(self, exercise_data):
        self.exercise = exercise_data
        self.analyzer = ExerciseAnalyzer()
        self.rep = 0
        self.step = 0
        self.last = -1
        self.hold = 0
        self.acc = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        lms, pose_lms = self.analyzer.get_landmarks(img)
        if lms is not None:
            mp_drawing.draw_landmarks(img, pose_lms, mp_pose.POSE_CONNECTIONS)
            angles = self.analyzer.get_angles(lms)

            best_i, best_a = 0, 0
            for i, s in enumerate(self.exercise["steps_data"]):
                a = self.analyzer.compare(angles, s["angles"])
                if a > best_a:
                    best_a, best_i = a, i

            self.acc = best_a
            if best_a > 70:
                exp = (self.last + 1) % len(self.exercise["steps_data"])
                if best_i == exp:
                    self.hold += 1
                    if self.hold >= 8:
                        self.last = best_i
                        self.step = best_i
                        self.hold = 0
                        if best_i == len(self.exercise["steps_data"]) - 1:
                            self.rep += 1

            cv2.putText(img, f"REPS: {self.rep}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==================== LOAD EXERCISES ====================
if "exercises" not in st.session_state:
    st.session_state.exercises = {}
    for f in DATA_DIR.glob("*.json"):
        st.session_state.exercises[f.stem] = json.load(open(f))

# ==================== UI ====================
st.title("🏃‍♂️ PhysioCheck")

if not st.session_state.exercises:
    st.warning("Train an exercise first.")
    st.stop()

ex = st.selectbox("Select exercise", list(st.session_state.exercises.keys()))
data = st.session_state.exercises[ex]

st.subheader("📹 Live Camera")

webrtc_streamer(
    key="physio",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=lambda: VideoProcessor(data),
    media_stream_constraints={"video": True, "audio": False},
)

st.markdown("---")
st.caption("PhysioCheck – Streamlit Cloud Ready ✅")
