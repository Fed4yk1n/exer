# app.py - PhysioCheck with LIVE CAMERA (WebRTC) — CLOUD SAFE

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
    initial_sidebar_state="expanded"
)

# ==================== CSS ====================
st.markdown("""
<style>
    .stApp { background: #0a0e27; }
    [data-testid="stSidebar"] { background: #111827; }
    h1, h2, h3 { color: #3b82f6 !important; }

    .exercise-card {
        background: #1f2937;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #374151;
        margin: 10px 0;
    }

    .stButton>button {
        background: #3b82f6;
        color: white;
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

# ==================== RTC CONFIG ====================
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ==================== HELPERS ====================
def img_to_b64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

def display_img(path, size="120px"):
    b64 = img_to_b64(path)
    if not b64:
        return
    st.markdown(f"""
    <div style="width:{size};height:{size};background:#111827;border:2px solid #374151;
                border-radius:10px;display:flex;align-items:center;justify-content:center;
                overflow:hidden;margin:8px auto;">
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
            min_tracking_confidence=0.5
        )

    def landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if res.pose_landmarks:
            lm = [[l.x, l.y, l.z, l.visibility] for l in res.pose_landmarks.landmark]
            return np.array(lm), res.pose_landmarks
        return None, None

    @staticmethod
    def angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        r = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        ang = abs(r * 180 / np.pi)
        return 360-ang if ang > 180 else ang

    def angles(self, l):
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

# ==================== VIDEO PROCESSOR (WEBRTC SAFE) ====================
class VideoProcessor:
    def __init__(self, exercise):
        self.exercise = exercise
        self.analyzer = ExerciseAnalyzer()
        self.rep = 0
        self.step = 0
        self.last = -1
        self.hold = 0
        self.acc = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        lm, pose_lm = self.analyzer.landmarks(img)

        if lm is not None:
            mp_drawing.draw_landmarks(img, pose_lm, mp_pose.POSE_CONNECTIONS)
            ang = self.analyzer.angles(lm)

            best_i, best_a = 0, 0
            for i, s in enumerate(self.exercise["steps_data"]):
                a = self.analyzer.compare(ang, s["angles"])
                if a > best_a:
                    best_a, best_i = a, i

            self.acc = best_a
            if best_a >= 70:
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
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0), 3)
            cv2.putText(img, f"STEP: {self.step+1}/{len(self.exercise['steps_data'])}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(img, f"MATCH: {self.acc:.0f}%",
                        (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==================== LOAD EXERCISES ====================
if "exercises" not in st.session_state:
    st.session_state.exercises = {}
    for f in DATA_DIR.glob("*.json"):
        st.session_state.exercises[f.stem] = json.load(open(f))

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### 🏃‍♂️ PhysioCheck")
    st.metric("Exercises", len(st.session_state.exercises))
    st.success("WebRTC Ready")

# ==================== MAIN ====================
st.title("🏃‍♂️ PhysioCheck")
tabs = st.tabs(["📹 Practice (Live)", "🎯 Train Exercise"])

# ==================== TRAIN TAB ====================
with tabs[1]:
    name = st.text_input("Exercise Name")
    diff = st.selectbox("Difficulty", ["Beginner","Intermediate","Advanced"])
    steps = st.number_input("Steps", 2, 6, 2)

    imgs = []
    cols = st.columns(min(4, steps))
    for i in range(steps):
        with cols[i % 4]:
            f = st.file_uploader(f"Step {i+1}", type=["jpg","png"], key=f"s{i}")
            if f:
                img = Image.open(f)
                imgs.append(img)
                img.save(STEPS_DIR / f"{name}_step_{i+1}.jpg")

    muscles = [m.lower() for m in ["Biceps","Triceps","Core"] if st.checkbox(m)]

    if st.button("Train", disabled=not(name and len(imgs)==steps)):
        analyzer = ExerciseAnalyzer()
        sd = []
        for img in imgs:
            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            lm,_ = analyzer.landmarks(bgr)
            sd.append({"angles": analyzer.angles(lm)})
        data = {
            "name": name,
            "difficulty": diff,
            "steps_data": sd,
            "step_images": [str(STEPS_DIR / f"{name}_step_{i+1}.jpg") for i in range(steps)],
            "muscle_targeting": {"primary": muscles, "secondary":[]}
        }
        json.dump(data, open(DATA_DIR / f"{name}.json","w"), indent=2)
        st.session_state.exercises[name] = data
        st.success("Exercise trained")

# ==================== PRACTICE TAB ====================
with tabs[0]:
    if not st.session_state.exercises:
        st.warning("Train an exercise first")
    else:
        ex = st.selectbox("Select Exercise", list(st.session_state.exercises))
        data = st.session_state.exercises[ex]

        col1, col2 = st.columns([3,1])

        with col2:
            for p in data["step_images"]:
                display_img(p)

        with col1:
            st.info("Click START and allow camera access")
            webrtc_streamer(
                key="physio",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=lambda: VideoProcessor(data),
                media_stream_constraints={"video": True, "audio": False},
            )

st.markdown("---")
st.caption("PhysioCheck — Streamlit Community Cloud compatible ✅")
