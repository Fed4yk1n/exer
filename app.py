# app.py - PhysioCheck with LIVE CAMERA (WebRTC)
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

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

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
        border: none;
        border-radius: 8px;
    }
    
    .success-box {
        background: #1f2937;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #10b981;
        color: #e5e7eb;
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
def get_image_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

def pil_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

def display_image_frame(src, size="150px"):
    if isinstance(src, str):
        b64 = get_image_base64(src)
    elif isinstance(src, Image.Image):
        b64 = pil_to_base64(src)
    else:
        return
    if b64:
        st.markdown(f"""
        <div style="width:{size};height:{size};background:#111827;border:2px solid #374151;
                    border-radius:10px;display:flex;align-items:center;justify-content:center;
                    overflow:hidden;margin:10px auto;">
            <img src="data:image/jpeg;base64,{b64}" style="max-width:100%;max-height:100%;object-fit:contain;"/>
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
    
    @staticmethod
    def calc_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        ang = np.abs(rad * 180 / np.pi)
        return 360 - ang if ang > 180 else ang
    
    def get_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if res.pose_landmarks:
            lms = [[l.x, l.y, l.z, l.visibility] for l in res.pose_landmarks.landmark]
            return np.array(lms), res.pose_landmarks
        return None, None
    
    def get_angles(self, lms):
        angles = {}
        try:
            angles['left_elbow'] = self.calc_angle(lms[11][:2], lms[13][:2], lms[15][:2])
            angles['right_elbow'] = self.calc_angle(lms[12][:2], lms[14][:2], lms[16][:2])
            angles['left_knee'] = self.calc_angle(lms[23][:2], lms[25][:2], lms[27][:2])
            angles['right_knee'] = self.calc_angle(lms[24][:2], lms[26][:2], lms[28][:2])
            angles['left_shoulder'] = self.calc_angle(lms[13][:2], lms[11][:2], lms[23][:2])
            angles['right_shoulder'] = self.calc_angle(lms[14][:2], lms[12][:2], lms[24][:2])
            angles['left_hip'] = self.calc_angle(lms[11][:2], lms[23][:2], lms[25][:2])
            angles['right_hip'] = self.calc_angle(lms[12][:2], lms[24][:2], lms[26][:2])
        except:
            pass
        return angles
    
    def process_image(self, img):
        if isinstance(img, Image.Image):
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        lms, _ = self.get_landmarks(img)
        if lms is not None:
            return {'landmarks': lms.tolist(), 'angles': self.get_angles(lms)}
        return None
    
    def compare(self, curr, ref, tol=20):
        if not curr or not ref:
            return 0
        total = correct = 0
        for j, a in ref.items():
            if j in curr:
                total += 1
                if abs(curr[j] - a) <= tol:
                    correct += 1
        return (correct / total * 100) if total else 0

# ==================== VIDEO PROCESSOR FOR WEBRTC ====================
class VideoProcessor:
    def __init__(self):
        self.analyzer = ExerciseAnalyzer()
        self.exercise_data = None
        self.current_step = 0
        self.rep_count = 0
        self.accuracy = 0
        self.last_step = -1
        self.hold_count = 0
        self.steps_done = []
    
    def set_exercise(self, data):
        self.exercise_data = data
        self.rep_count = 0
        self.current_step = 0
        self.last_step = -1
        self.steps_done = []
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        if not self.exercise_data:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        lms, pose_lms = self.analyzer.get_landmarks(img)
        
        if lms is not None:
            # Draw skeleton
            mp_drawing.draw_landmarks(
                img, pose_lms, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(59, 130, 246), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(59, 130, 246), thickness=2)
            )
            
            curr_angles = self.analyzer.get_angles(lms)
            
            # Find best step
            best_idx, best_acc = 0, 0
            for i, s in enumerate(self.exercise_data['steps_data']):
                acc = self.analyzer.compare(curr_angles, s['angles'])
                if acc > best_acc:
                    best_acc = acc
                    best_idx = i
            
            self.accuracy = best_acc
            
            # Rep counting logic
            if best_acc >= 70:
                expected = (self.last_step + 1) % len(self.exercise_data['steps_data'])
                if best_idx == expected:
                    self.hold_count += 1
                    if self.hold_count >= 8:
                        self.last_step = best_idx
                        self.current_step = best_idx
                        self.hold_count = 0
                        if best_idx not in self.steps_done:
                            self.steps_done.append(best_idx)
                        if best_idx == len(self.exercise_data['steps_data']) - 1:
                            if len(self.steps_done) == len(self.exercise_data['steps_data']):
                                self.rep_count += 1
                            self.steps_done = []
                else:
                    self.hold_count = 0
            
            # Draw info
            h, w = img.shape[:2]
            cv2.putText(img, f'REPS: {self.rep_count}', (15, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (59, 130, 246), 3)
            cv2.putText(img, f'Step {self.current_step+1}/{len(self.exercise_data["steps_data"])}', 
                       (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (16, 185, 129), 2)
            cv2.putText(img, f'Match: {self.accuracy:.0f}%', 
                       (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw muscle dots
            primary = self.exercise_data.get('muscle_targeting', {}).get('primary', [])
            if 'biceps' in primary:
                for idx in [13, 14]:
                    if lms[idx][3] > 0.5:
                        x, y = int(lms[idx][0] * w), int(lms[idx][1] * h)
                        cv2.circle(img, (x, y), 12, (0, 0, 255), -1)
                        cv2.circle(img, (x, y), 12, (255, 255, 255), 2)
        else:
            cv2.putText(img, 'NO POSE DETECTED', (15, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Global processor instance
if 'processor' not in st.session_state:
    st.session_state.processor = VideoProcessor()

# ==================== SESSION STATE ====================
if 'exercises' not in st.session_state:
    st.session_state.exercises = {}
    for f in DATA_DIR.glob("*.json"):
        try:
            st.session_state.exercises[f.stem] = json.load(open(f))
        except:
            pass

if 'selected_exercise' not in st.session_state:
    st.session_state.selected_exercise = None

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### 🏃‍♂️ PhysioCheck")
    st.markdown("---")
    st.metric("Exercises", len(st.session_state.exercises))
    st.markdown("---")
    if WEBRTC_AVAILABLE:
        st.success("✅ Live Camera Ready")
    else:
        st.error("❌ WebRTC not available")

# ==================== MAIN ====================
st.markdown("# 🏃‍♂️ PhysioCheck")
st.markdown("AI-Powered Physiotherapy with **Live Camera**")

tab1, tab2 = st.tabs(["📹 Practice (Live)", "🎯 Train Exercise"])

# ==================== TRAIN TAB ====================
with tab2:
    st.markdown("## Train New Exercise")
    
    name = st.text_input("Exercise Name", placeholder="e.g., Bicep Curl")
    difficulty = st.selectbox("Difficulty", ["Beginner", "Intermediate", "Advanced"])
    num_steps = st.number_input("Number of Steps", 2, 6, 2)
    
    st.markdown("### Upload Step Images")
    
    step_imgs = []
    cols = st.columns(min(4, num_steps))
    for i in range(num_steps):
        with cols[i % 4]:
            f = st.file_uploader(f"Step {i+1}", type=['png','jpg','jpeg'], key=f"s{i}")
            if f:
                img = Image.open(f)
                step_imgs.append(img)
                display_image_frame(img, "130px")
    
    st.markdown("### Muscles")
    MUSCLES = ['Biceps', 'Triceps', 'Shoulders', 'Chest', 'Core', 'Quadriceps', 'Glutes']
    primary = [m.lower() for m in MUSCLES if st.checkbox(m, key=f"m_{m}")]
    
    if st.button("🎯 Train", disabled=not(name and len(step_imgs)==num_steps and primary)):
        with st.spinner("Processing..."):
            analyzer = ExerciseAnalyzer()
            steps_data, paths = [], []
            ok = True
            
            for i, img in enumerate(step_imgs):
                arr = np.array(img)
                if arr.shape[-1] == 4:
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                
                pose = analyzer.process_image(bgr)
                if pose:
                    steps_data.append(pose)
                    p = STEPS_DIR / f"{name}_step_{i+1}.jpg"
                    img.save(p, "JPEG")
                    paths.append(str(p))
                else:
                    st.error(f"No pose in Step {i+1}")
                    ok = False
                    break
            
            if ok:
                data = {
                    'name': name,
                    'difficulty': difficulty,
                    'num_steps': num_steps,
                    'steps_data': steps_data,
                    'step_images': paths,
                    'created_at': datetime.now().isoformat(),
                    'muscle_targeting': {'primary': primary, 'secondary': []}
                }
                json.dump(data, open(DATA_DIR / f"{name}.json", 'w'), indent=2)
                st.session_state.exercises[name] = data
                st.success(f"✅ {name} trained!")
                st.balloons()

# ==================== PRACTICE TAB ====================
with tab1:
    st.markdown("## Practice with Live Camera")
    
    if not st.session_state.exercises:
        st.warning("No exercises. Train one first!")
    elif not WEBRTC_AVAILABLE:
        st.error("WebRTC not available. Camera won't work.")
    else:
        # Exercise selector
        ex_name = st.selectbox("Select Exercise", list(st.session_state.exercises.keys()))
        
        if ex_name:
            ex_data = st.session_state.exercises[ex_name]
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.markdown("### Steps")
                for i, p in enumerate(ex_data.get('step_images', [])):
                    if os.path.exists(p):
                        st.markdown(f"**Step {i+1}**")
                        display_image_frame(p, "100px")
            
            with col1:
                st.markdown("### 📹 Live Camera")
                st.info("👆 Click **START** below, then allow camera access")
                
                # Set exercise data for processor
                st.session_state.processor.set_exercise(ex_data)
                
                # WebRTC streamer
                webrtc_streamer(
                    key="physio",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=RTC_CONFIGURATION,
                    video_processor_factory=lambda: st.session_state.processor,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )
                
                # Show current stats
                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                c1.metric("Reps", st.session_state.processor.rep_count)
                c2.metric("Step", f"{st.session_state.processor.current_step + 1}/{ex_data['num_steps']}")
                c3.metric("Match", f"{st.session_state.processor.accuracy:.0f}%")

st.markdown("---")
st.markdown("<center><small>PhysioCheck - Live AI Physiotherapy</small></center>", unsafe_allow_html=True)
