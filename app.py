# app.py - PhysioCheck: AI-Powered Physiotherapy Assistant (CLOUD-READY VERSION)
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime
import time
import gc
from collections import deque
import base64
from PIL import Image
import io
import threading


# WebRTC imports
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="PhysioCheck - AI Physiotherapy",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS - DARK MINIMALIST THEME ====================
def load_css():
    st.markdown("""
    <style>
        /* Dark Background */
        .stApp {
            background: #0a0e27;
        }
        
        [data-testid="stSidebar"] {
            background: #111827;
        }
        
        /* Clean Headers - No Glow */
        h1, h2, h3 {
            color: #3b82f6 !important;
            font-weight: 600 !important;
            text-shadow: none !important;
        }
        
        /* Minimalist Cards */
        .exercise-card {
            background: #1f2937;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #374151;
            margin: 10px 0;
            transition: border-color 0.3s ease;
        }
        
        .exercise-card:hover {
            border-color: #3b82f6;
        }
        
        /* Clean Buttons */
        .stButton>button {
            background: #3b82f6;
            color: white;
            font-weight: 500;
            border: none;
            border-radius: 8px;
            padding: 10px 24px;
            transition: background 0.3s ease;
        }
        
        .stButton>button:hover {
            background: #2563eb;
        }
        
        /* Input Fields */
        .stTextInput>div>div>input, .stSelectbox>div>div>select, .stNumberInput>div>div>input {
            background-color: #1f2937;
            color: #e5e7eb;
            border: 1px solid #374151;
            border-radius: 6px;
        }
        
        /* Progress Bar */
        .stProgress > div > div > div {
            background: #3b82f6;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            color: #3b82f6 !important;
            font-size: 2rem !important;
        }
        
        /* File Uploader */
        [data-testid="stFileUploader"] {
            background-color: #1f2937;
            border: 2px dashed #374151;
            border-radius: 8px;
            padding: 15px;
        }
        
        /* Alert Boxes */
        .success-box {
            background: #1f2937;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #10b981;
            color: #e5e7eb;
            margin: 10px 0;
        }
        
        .error-box {
            background: #1f2937;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #ef4444;
            color: #e5e7eb;
            margin: 10px 0;
        }
        
        .warning-box {
            background: #1f2937;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #f59e0b;
            color: #e5e7eb;
            margin: 10px 0;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #111827;
            border-radius: 8px;
            padding: 4px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #1f2937;
            color: #9ca3af;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            background: #3b82f6;
            color: white;
        }
        
        /* Fixed Frame Container for Images/Videos */
        .fixed-frame {
            width: 200px;
            height: 200px;
            background: #111827;
            border: 2px solid #374151;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            margin: 10px auto;
        }
        
        .fixed-frame img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
        
        .fixed-frame-large {
            width: 280px;
            height: 280px;
            background: #111827;
            border: 2px solid #374151;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            margin: 10px auto;
        }
        
        .fixed-frame-large img, .fixed-frame-large video {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
        
        .muscle-category {
            background: #1f2937;
            padding: 10px;
            border-radius: 6px;
            margin: 5px 0;
            border: 1px solid #374151;
        }
        
        /* Video Container */
        .video-container {
            position: relative;
            width: 100%;
            max-width: 300px;
            border-radius: 8px;
            overflow: hidden;
            margin: 0 auto;
        }
        
        .video-container video {
            width: 100%;
            border-radius: 8px;
        }
        
        /* Upload Preview Container */
        .upload-preview {
            width: 180px;
            height: 180px;
            background: #111827;
            border: 2px solid #374151;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            margin: 10px auto;
        }
        
        .upload-preview img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
    </style>
    """, unsafe_allow_html=True)

load_css()

# ==================== MEDIAPIPE INITIALIZATION ====================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ==================== WEBRTC CONFIGURATION ====================
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ==================== DIRECTORY SETUP ====================
DATA_DIR = Path("exercise_data")
DATA_DIR.mkdir(exist_ok=True)
STEPS_DIR = DATA_DIR / "steps"
STEPS_DIR.mkdir(exist_ok=True)

# ==================== HELPER FUNCTIONS ====================
def get_video_base64(video_path):
    """Convert video to base64 for HTML5 player"""
    try:
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            return None
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        if len(video_bytes) == 0:
            print(f"Empty video file: {video_path}")
            return None
        video_base64 = base64.b64encode(video_bytes).decode()
        return video_base64
    except Exception as e:
        print(f"Error encoding video: {e}")
        return None

def display_looping_video(video_path, max_width="300px"):
    """Display HTML5 video with loop in fixed container"""
    video_base64 = get_video_base64(video_path)
    if video_base64:
        video_html = f"""
        <div class="video-container" style="max-width: {max_width};">
            <video autoplay loop muted playsinline style="border-radius: 8px; width: 100%;">
                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        """
        st.markdown(video_html, unsafe_allow_html=True)
    else:
        st.error("Could not load video")

def get_image_base64(image_path):
    """Convert image to base64 with safety checks"""
    try:
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        if len(image_bytes) == 0:
            print(f"Empty image file: {image_path}")
            return None
        image_base64 = base64.b64encode(image_bytes).decode()
        return image_base64
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def pil_to_base64(pil_image):
    """Convert PIL image to base64"""
    try:
        buffered = io.BytesIO()
        if pil_image.mode == "RGBA":
            pil_image = pil_image.convert("RGB")
        elif pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        pil_image.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        return img_base64
    except Exception as e:
        print(f"Error converting PIL to base64: {e}")
        return None

def display_image_in_frame(image_source, size="180px"):
    """Display image in a fixed square frame"""
    img_base64 = None
    if isinstance(image_source, str):
        img_base64 = get_image_base64(image_source)
    elif isinstance(image_source, Image.Image):
        img_base64 = pil_to_base64(image_source)
    
    if img_base64:
        html = f"""
        <div style="width: {size}; height: {size}; background: #111827; 
                    border: 2px solid #374151; border-radius: 10px; 
                    display: flex; align-items: center; justify-content: center; 
                    overflow: hidden; margin: 10px auto;">
            <img src="data:image/jpeg;base64,{img_base64}" 
                 style="max-width: 100%; max-height: 100%; object-fit: contain;" />
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="width: {size}; height: {size}; background: #1f2937; 
                    border: 2px solid #ef4444; border-radius: 10px; 
                    display: flex; align-items: center; justify-content: center; 
                    margin: 10px auto; color: #ef4444; font-size: 0.8rem;">
            Image Error
        </div>
        """, unsafe_allow_html=True)

# ==================== MUSCLE VISUALIZER CLASS ====================
class MuscleVisualizer:
    """Display targeted muscles with elegant minimalistic highlights"""
    
    def __init__(self):
        self.muscle_regions = {
            'biceps': {
                'left': {'points': [11, 13], 'position': 0.4},
                'right': {'points': [12, 14], 'position': 0.4}
            },
            'triceps': {
                'left': {'points': [11, 13], 'position': 0.6},
                'right': {'points': [12, 14], 'position': 0.6}
            },
            'forearms': {
                'left': {'points': [13, 15], 'position': 0.5},
                'right': {'points': [14, 16], 'position': 0.5}
            },
            'shoulders': {
                'left': {'points': [11], 'position': 0},
                'right': {'points': [12], 'position': 0}
            },
            'chest': {
                'center': {'points': [11, 12], 'position': 0.5}
            },
            'upper_back': {
                'center': {'points': [11, 12], 'position': 0.5}
            },
            'lower_back': {
                'center': {'points': [23, 24], 'position': 0.5}
            },
            'quadriceps': {
                'left': {'points': [23, 25], 'position': 0.4},
                'right': {'points': [24, 26], 'position': 0.4}
            },
            'hamstrings': {
                'left': {'points': [23, 25], 'position': 0.6},
                'right': {'points': [24, 26], 'position': 0.6}
            },
            'glutes': {
                'left': {'points': [23], 'position': 0},
                'right': {'points': [24], 'position': 0}
            },
            'calves': {
                'left': {'points': [25, 27], 'position': 0.5},
                'right': {'points': [26, 28], 'position': 0.5}
            },
            'core': {
                'center': {'points': [11, 12, 23, 24], 'position': 0}
            },
            'abs': {
                'center': {'points': [11, 12, 23, 24], 'position': 0}
            },
            'obliques': {
                'left': {'points': [11, 23], 'position': 0.5},
                'right': {'points': [12, 24], 'position': 0.5}
            }
        }
    
    def get_active_muscles(self, exercise_data):
        if 'muscle_targeting' in exercise_data:
            return exercise_data['muscle_targeting']
        return {'primary': [], 'secondary': []}
    
    def draw_muscle_highlights(self, frame, landmarks, active_muscles):
        if landmarks is None:
            return frame
        
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        primary_muscles = active_muscles.get('primary', [])
        secondary_muscles = active_muscles.get('secondary', [])
        
        for muscle in primary_muscles:
            if muscle in self.muscle_regions:
                self._draw_elegant_muscle(
                    overlay, landmarks, self.muscle_regions[muscle],
                    primary_color=(0, 0, 255),
                    glow_color=(0, 0, 180),
                    w=w, h=h, is_primary=True
                )
        
        for muscle in secondary_muscles:
            if muscle in self.muscle_regions:
                self._draw_elegant_muscle(
                    overlay, landmarks, self.muscle_regions[muscle],
                    primary_color=(0, 140, 255),
                    glow_color=(0, 100, 180),
                    w=w, h=h, is_primary=False
                )
        
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        return frame
    
    def _draw_elegant_muscle(self, frame, landmarks, region_config, primary_color, glow_color, w, h, is_primary):
        for side, config in region_config.items():
            points_indices = config['points']
            position = config['position']
            center = self._get_muscle_center(landmarks, points_indices, position, w, h)
            if center is None:
                continue
            cx, cy = center
            radius = 12 if is_primary else 9
            glow_radius = 20 if is_primary else 15
            for i in range(3):
                glow_r = glow_radius + (i * 4)
                alpha = 0.15 - (i * 0.04)
                temp_overlay = frame.copy()
                cv2.circle(temp_overlay, (cx, cy), glow_r, glow_color, -1)
                cv2.addWeighted(temp_overlay, alpha, frame, 1 - alpha, 0, frame)
            cv2.circle(frame, (cx, cy), radius, primary_color, -1)
            inner_radius = max(3, radius - 4)
            bright_color = tuple(min(255, c + 80) for c in primary_color)
            cv2.circle(frame, (cx, cy), inner_radius, bright_color, -1)
            cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 1)
    
    def _get_muscle_center(self, landmarks, points_indices, position, w, h):
        if len(points_indices) == 1:
            idx = points_indices[0]
            if idx < len(landmarks) and landmarks[idx][3] > 0.5:
                x = int(landmarks[idx][0] * w)
                y = int(landmarks[idx][1] * h)
                return (x, y)
        elif len(points_indices) == 2:
            idx1, idx2 = points_indices[0], points_indices[1]
            if (idx1 < len(landmarks) and landmarks[idx1][3] > 0.5 and
                idx2 < len(landmarks) and landmarks[idx2][3] > 0.5):
                x1, y1 = landmarks[idx1][0] * w, landmarks[idx1][1] * h
                x2, y2 = landmarks[idx2][0] * w, landmarks[idx2][1] * h
                cx = int(x1 + (x2 - x1) * position)
                cy = int(y1 + (y2 - y1) * position)
                return (cx, cy)
        elif len(points_indices) == 4:
            valid_points = []
            for idx in points_indices:
                if idx < len(landmarks) and landmarks[idx][3] > 0.5:
                    x = landmarks[idx][0] * w
                    y = landmarks[idx][1] * h
                    valid_points.append((x, y))
            if len(valid_points) >= 3:
                cx = int(np.mean([p[0] for p in valid_points]))
                cy = int(np.mean([p[1] for p in valid_points]))
                return (cx, cy)
        return None

# ==================== STEP-BASED EXERCISE ANALYZER ====================
class StepBasedExerciseAnalyzer:
    """Analyze exercise with step-by-step pose matching"""
    def __init__(self, model_complexity=1):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.fps_queue = deque(maxlen=30)
        self.last_time = time.time()
    
    def __del__(self):
        if hasattr(self, 'pose'):
            self.pose.close()
    
    def get_fps(self):
        current_time = time.time()
        fps = 1 / (current_time - self.last_time) if (current_time - self.last_time) > 0 else 0
        self.last_time = current_time
        self.fps_queue.append(fps)
        return np.mean(self.fps_queue) if self.fps_queue else 0
    
    @staticmethod
    def calculate_angle(a, b, c):
        try:
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)
            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians * 180.0 / np.pi)
            if angle > 180.0:
                angle = 360 - angle
            return angle
        except:
            return 0.0
    
    def extract_landmarks(self, frame):
        try:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = self.pose.process(image_rgb)
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
                return np.array(landmarks), results.pose_landmarks
            return None, None
        except Exception as e:
            print(f"Error extracting landmarks: {e}")
            return None, None
    
    def calculate_key_angles(self, landmarks):
        angles = {}
        try:
            angles['left_elbow'] = self.calculate_angle(
                [landmarks[11][0], landmarks[11][1]],
                [landmarks[13][0], landmarks[13][1]],
                [landmarks[15][0], landmarks[15][1]]
            )
            angles['right_elbow'] = self.calculate_angle(
                [landmarks[12][0], landmarks[12][1]],
                [landmarks[14][0], landmarks[14][1]],
                [landmarks[16][0], landmarks[16][1]]
            )
            angles['left_knee'] = self.calculate_angle(
                [landmarks[23][0], landmarks[23][1]],
                [landmarks[25][0], landmarks[25][1]],
                [landmarks[27][0], landmarks[27][1]]
            )
            angles['right_knee'] = self.calculate_angle(
                [landmarks[24][0], landmarks[24][1]],
                [landmarks[26][0], landmarks[26][1]],
                [landmarks[28][0], landmarks[28][1]]
            )
            angles['left_shoulder'] = self.calculate_angle(
                [landmarks[13][0], landmarks[13][1]],
                [landmarks[11][0], landmarks[11][1]],
                [landmarks[23][0], landmarks[23][1]]
            )
            angles['right_shoulder'] = self.calculate_angle(
                [landmarks[14][0], landmarks[14][1]],
                [landmarks[12][0], landmarks[12][1]],
                [landmarks[24][0], landmarks[24][1]]
            )
            angles['left_hip'] = self.calculate_angle(
                [landmarks[11][0], landmarks[11][1]],
                [landmarks[23][0], landmarks[23][1]],
                [landmarks[25][0], landmarks[25][1]]
            )
            angles['right_hip'] = self.calculate_angle(
                [landmarks[12][0], landmarks[12][1]],
                [landmarks[24][0], landmarks[24][1]],
                [landmarks[26][0], landmarks[26][1]]
            )
        except Exception as e:
            print(f"Error calculating angles: {e}")
        return angles
    
    def process_step_image(self, image):
        try:
            if isinstance(image, Image.Image):
                image = np.array(image)
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif len(image.shape) == 3 and image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                elif len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            landmarks, _ = self.extract_landmarks(image)
            if landmarks is not None:
                angles = self.calculate_key_angles(landmarks)
                return {'landmarks': landmarks.tolist(), 'angles': angles}
            return None
        except Exception as e:
            print(f"Error processing step image: {e}")
            return None
    
    def compare_with_step(self, current_angles, step_angles, tolerance=20):
        if not current_angles or not step_angles:
            return 0.0
        total_joints = 0
        correct_joints = 0
        for joint, ref_angle in step_angles.items():
            if joint in current_angles:
                total_joints += 1
                diff = abs(current_angles[joint] - ref_angle)
                if diff <= tolerance:
                    correct_joints += 1
        return (correct_joints / total_joints * 100) if total_joints > 0 else 0

# ==================== STEP-BASED REP COUNTER ====================
class StepBasedRepCounter:
    def __init__(self, steps_data):
        self.steps_data = steps_data
        self.num_steps = len(steps_data)
        self.current_step = 0
        self.rep_count = 0
        self.step_completion_threshold = 70
        self.step_hold_frames = 8
        self.current_step_hold_count = 0
        self.last_completed_step = -1
        self.steps_completed_this_rep = []
    
    def detect_step_and_count(self, current_angles, analyzer):
        if not self.steps_data or not current_angles:
            return 0, self.rep_count, 0
        
        step_accuracies = []
        for step_data in self.steps_data:
            accuracy = analyzer.compare_with_step(current_angles, step_data['angles'])
            step_accuracies.append(accuracy)
        
        best_step_idx = int(np.argmax(step_accuracies))
        best_accuracy = step_accuracies[best_step_idx]
        
        if best_accuracy >= self.step_completion_threshold:
            expected_step = (self.last_completed_step + 1) % self.num_steps
            if best_step_idx == expected_step:
                self.current_step_hold_count += 1
                if self.current_step_hold_count >= self.step_hold_frames:
                    self.last_completed_step = best_step_idx
                    self.current_step = best_step_idx
                    self.current_step_hold_count = 0
                    if best_step_idx not in self.steps_completed_this_rep:
                        self.steps_completed_this_rep.append(best_step_idx)
                    if best_step_idx == self.num_steps - 1:
                        if len(self.steps_completed_this_rep) == self.num_steps:
                            self.rep_count += 1
                            print(f"[REP] Completed! Total: {self.rep_count}")
                        self.steps_completed_this_rep = []
            else:
                self.current_step_hold_count = 0
        else:
            self.current_step_hold_count = 0
        
        return self.current_step, self.rep_count, best_accuracy

# ==================== VIDEO PROCESSOR FOR WEBRTC ====================
class VideoProcessor:
    """Process video frames from browser camera in real-time"""

    def __init__(self, exercise_data, model_complexity=None):
        if model_complexity is None:
            model_complexity = 1

        self.exercise_data = exercise_data
        self.analyzer = StepBasedExerciseAnalyzer(model_complexity=model_complexity)
        self.counter = StepBasedRepCounter(exercise_data['steps_data'])
        self.muscle_viz = MuscleVisualizer()
        self.active_muscles = self.muscle_viz.get_active_muscles(exercise_data)
        self.lock = threading.Lock()

        # Current stats
        self.current_step = 0
        self.rep_count = 0
        self.step_accuracy = 0
        self.pose_detected = False

    
    def recv(self, frame):
        """Process each video frame from camera"""
        img = frame.to_ndarray(format="bgr24")
        
        # Flip horizontally for mirror effect
        img = cv2.flip(img, 1)
        
        # Extract pose landmarks
        landmarks, pose_landmarks = self.analyzer.extract_landmarks(img)
        
        if landmarks is not None:
            self.pose_detected = True
            
            # Draw muscle highlights
            img = self.muscle_viz.draw_muscle_highlights(img, landmarks, self.active_muscles)
            
            # Draw skeleton overlay
            mp_drawing.draw_landmarks(
                img,
                pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(59, 130, 246), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(59, 130, 246), thickness=2)
            )
            
            # Calculate angles and detect steps
            current_angles = self.analyzer.calculate_key_angles(landmarks)
            
            with self.lock:
                self.current_step, self.rep_count, self.step_accuracy = \
                    self.counter.detect_step_and_count(current_angles, self.analyzer)
            
            # Draw info on frame
            h, w = img.shape[:2]
            
            # Rep count (large, top-left)
            cv2.putText(img, f'Reps: {self.rep_count}', (15, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (59, 130, 246), 3)
            
            # Current step
            cv2.putText(img, f'Step: {self.current_step + 1}/{self.exercise_data["num_steps"]}',
                        (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (16, 185, 129), 2)
            
            # Accuracy with color coding
            accuracy_color = (16, 185, 129) if self.step_accuracy >= 70 else (239, 68, 68)
            cv2.putText(img, f'Match: {self.step_accuracy:.0f}%',
                        (15, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, accuracy_color, 2)
            
        else:
            self.pose_detected = False
            h, w = img.shape[:2]
            
            # No pose detected message
            cv2.putText(img, 'NO POSE DETECTED', (w//2 - 200, h//2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (239, 68, 68), 3)
            cv2.putText(img, 'Stand in view of camera', (w//2 - 180, h//2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (156, 163, 175), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==================== SESSION STATE INITIALIZATION ====================


# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### 🏃‍♂️ PhysioCheck")
    st.markdown("---")
    st.markdown("**AI-Powered Physiotherapy**")
    st.markdown("---")
    
    total_exercises = len(st.session_state.exercises)
    st.metric("Exercises", total_exercises)
    
    st.markdown("---")
        st.markdown("#### Settings")

    st.selectbox(
        "Model Quality",
        options=[0, 1, 2],
        format_func=lambda x: ["Lite (Fast)", "Balanced", "Accurate (Slow)"][x],
        index=1,
        key="model_complexity"
    )
    
   
    st.session_state.show_skeleton = st.checkbox(
        "Show Skeleton", 
        value=st.session_state.show_skeleton
    )

# ==================== MAIN APP ====================
st.markdown("# PhysioCheck")
st.markdown("AI-Powered Home Physiotherapy Assistant")

tab1, tab2 = st.tabs(["Practice", "Train New Exercise"])

# ==================== TRAIN TAB ====================
with tab2:
    st.markdown("## Train New Exercise")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        exercise_name = st.text_input("Exercise Name", placeholder="e.g., Bicep Curls")
    with col2:
        difficulty = st.selectbox("Difficulty", ["Beginner", "Intermediate", "Advanced"])
    
    st.markdown("---")
    st.markdown("### Exercise Steps")
    
    num_steps = st.number_input(
        "How many steps in this exercise?",
        min_value=2,
        max_value=6,
        value=st.session_state.num_steps,
        help="E.g., Bicep curl = 2 steps (down, up)"
    )
    st.session_state.num_steps = num_steps
    
    st.markdown("#### Upload Step Images (in order)")
    st.info("📸 Upload clear images showing the correct pose for each step")
    
    step_images = []
    step_cols = st.columns(min(4, num_steps))
    for i in range(num_steps):
        with step_cols[i % 4]:
            st.markdown(f"**Step {i+1}**")
            step_img = st.file_uploader(
                f"Step {i+1}",
                type=['png', 'jpg', 'jpeg'],
                key=f"step_img_{i}",
                label_visibility="collapsed"
            )
            if step_img:
                img = Image.open(step_img)
                step_images.append(img)
                display_image_in_frame(img, size="150px")
            else:
                st.markdown("""
                <div style="width: 150px; height: 150px; background: #111827; 
                            border: 2px dashed #374151; border-radius: 10px; 
                            display: flex; align-items: center; justify-content: center; 
                            margin: 10px auto; color: #6b7280; font-size: 0.8rem;">
                    Upload image
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Reference Video (Optional)")
    
    vid_col1, vid_col2 = st.columns([1, 2])
    with vid_col1:
        uploaded_video = st.file_uploader(
            "Upload Reference Video",
            type=['mp4', 'avi', 'mov'],
            help="Optional: Video showing complete exercise"
        )
    with vid_col2:
        if uploaded_video:
            st.markdown("<div style='max-width: 280px; margin: 10px 0;'>", unsafe_allow_html=True)
            st.video(uploaded_video)
            st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Select Muscles Targeted")
    
    AVAILABLE_MUSCLES = {
        'Upper Body': {
            'Arms': ['Biceps', 'Triceps', 'Forearms'],
            'Shoulders & Back': ['Shoulders', 'Upper Back', 'Lower Back'],
            'Chest': ['Chest']
        },
        'Lower Body': {
            'Legs': ['Quadriceps', 'Hamstrings', 'Calves'],
            'Glutes': ['Glutes']
        },
        'Core': {
            'Torso': ['Abs', 'Core', 'Obliques']
        }
    }
    
    st.markdown("#### Primary Muscles")
    primary_selected = []
    primary_cols = st.columns(3)
    col_idx = 0
    for main_category, subcategories in AVAILABLE_MUSCLES.items():
        with primary_cols[col_idx % 3]:
            st.markdown(f"**{main_category}**")
            for subcat, muscles in subcategories.items():
                for muscle in muscles:
                    key = muscle.lower().replace(' ', '_')
                    if st.checkbox(muscle, key=f"primary_{key}"):
                        primary_selected.append(key)
        col_idx += 1
    
    st.markdown("#### Secondary Muscles")
    secondary_selected = []
    secondary_cols = st.columns(3)
    col_idx = 0
    for main_category, subcategories in AVAILABLE_MUSCLES.items():
        with secondary_cols[col_idx % 3]:
            st.markdown(f"**{main_category}**")
            for subcat, muscles in subcategories.items():
                for muscle in muscles:
                    key = muscle.lower().replace(' ', '_')
                    if key not in primary_selected:
                        if st.checkbox(muscle, key=f"secondary_{key}"):
                            secondary_selected.append(key)
        col_idx += 1
    
    st.markdown("---")
    
    can_train = exercise_name and len(step_images) == num_steps and (primary_selected or secondary_selected)
    if not can_train:
        st.warning(f"Please enter name, upload all {num_steps} step images, and select muscles")
    
    if st.button("Train Exercise", disabled=not can_train, type="primary"):
        try:
            with st.spinner("Processing step images..."):
                analyzer = StepBasedExerciseAnalyzer(model_complexity=1)
                steps_data = []
                step_image_paths = []
                all_steps_valid = True
                
                for i, step_img in enumerate(step_images):
                    try:
                        img_array = np.array(step_img)
                        if len(img_array.shape) == 2:
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        step_pose = analyzer.process_step_image(img_bgr)
                        
                        if step_pose:
                            steps_data.append(step_pose)
                            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
                            step_img_path = STEPS_DIR / f"{exercise_name}_step_{i+1}_{timestamp}.jpg"
                            if step_img.mode != "RGB":
                                step_img = step_img.convert("RGB")
                            step_img.save(step_img_path, "JPEG", quality=90)
                            if os.path.exists(step_img_path) and os.path.getsize(step_img_path) > 0:
                                step_image_paths.append(str(step_img_path))
                                st.success(f"✓ Step {i+1} processed successfully")
                            else:
                                st.error(f"Failed to save Step {i+1} image")
                                all_steps_valid = False
                                break
                        else:
                            st.error(f"Could not detect pose in Step {i+1}. Please upload a clearer image with a visible person.")
                            all_steps_valid = False
                            break
                    except Exception as img_error:
                        st.error(f"Error processing Step {i+1}: {str(img_error)}")
                        all_steps_valid = False
                        break
                
                if all_steps_valid and len(steps_data) == num_steps:
                    exercise_data = {
                        'name': exercise_name,
                        'difficulty': difficulty,
                        'num_steps': num_steps,
                        'steps_data': steps_data,
                        'step_images': step_image_paths,
                        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'muscle_targeting': {
                            'primary': primary_selected,
                            'secondary': secondary_selected
                        }
                    }
                    if uploaded_video:
                        video_save_path = DATA_DIR / f"{exercise_name}.mp4"
                        with open(video_save_path, 'wb') as f:
                            uploaded_video.seek(0)
                            f.write(uploaded_video.read())
                        exercise_data['video_path'] = str(video_save_path)
                    json_path = DATA_DIR / f"{exercise_name}.json"
                    with open(json_path, 'w') as f:
                        json.dump(exercise_data, f, indent=2)
                    st.session_state.exercises[exercise_name] = exercise_data
                    st.markdown(f"""
                    <div class='success-box'>
                        <h3>✓ Exercise Trained!</h3>
                        <p><strong>{exercise_name}</strong> - {num_steps} steps</p>
                        <p>Images saved: {len(step_image_paths)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.error("Training failed. Please check all step images and try again.")
        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

# ==================== PRACTICE TAB ====================
with tab1:
    st.markdown("## Practice Exercises")
    
    if not st.session_state.exercises:
        st.markdown("""
        <div class='warning-box'>
            <h3>No Exercises Available</h3>
            <p>Go to <strong>Train New Exercise</strong> tab to create your first exercise</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Display exercise cards
        exercise_names = list(st.session_state.exercises.keys())
        cols = st.columns(3)
        
        for idx, ex_name in enumerate(exercise_names):
            with cols[idx % 3]:
                ex_data = st.session_state.exercises[ex_name]
                muscles = ex_data.get('muscle_targeting', {})
                primary = muscles.get('primary', [])
                muscle_str = ", ".join([m.replace('_', ' ').title() for m in primary[:2]])
                if len(primary) > 2:
                    muscle_str += f" +{len(primary)-2}"
                
                st.markdown(f"""
                <div class='exercise-card'>
                    <h3>{ex_name}</h3>
                    <p>{ex_data['difficulty']} | {ex_data.get('num_steps', 0)} steps</p>
                    <p style="color: #ef4444;">{muscle_str if muscle_str else 'No muscles'}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Start {ex_name}", key=f"start_{idx}"):
                    st.session_state.selected_exercise = ex_name
        
        # Selected exercise view
        if st.session_state.selected_exercise:
            st.markdown("---")
            selected_exercise = st.session_state.selected_exercise
            st.markdown(f"## 🎯 {selected_exercise}")
            exercise_data = st.session_state.exercises[selected_exercise]
            
            col1, col2 = st.columns([3, 1])
            
            # Right column - Reference video
            with col2:
                st.markdown("### 📹 Reference")
                if 'video_path' in exercise_data and os.path.exists(exercise_data['video_path']):
                    display_looping_video(exercise_data['video_path'], max_width="280px")
                else:
                    st.info("No reference video")
                
                st.markdown("---")
                
                if st.button("🔙 Back to Exercises", key="back_btn", use_container_width=True):
                    st.session_state.selected_exercise = None
                    st.rerun()
            
            # Left column - Live camera
            with col1:
                st.markdown("### 📷 Live Camera Analysis")
                
                st.info("👇 Click **START** below and allow camera access in your browser")
                
                # Real-time metrics
                met_col1, met_col2, met_col3 = st.columns(3)
                rep_metric = met_col1.empty()
                step_metric = met_col2.empty()
                accuracy_metric = met_col3.empty()
                
                # WebRTC video streamer
                ctx = webrtc_streamer(
                    key=f"physiocheck-{selected_exercise}",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=RTC_CONFIGURATION,
                    video_processor_factory=lambda: VideoProcessor(
                        exercise_data, 
                        st.session_state.get("model_complexity", 1)
                    ),
                    media_stream_constraints={
                        "video": {
                            "width": {"ideal": 1280, "max": 1920},
                            "height": {"ideal": 720, "max": 1080},
                            "facingMode": "user"  # Front camera on mobile
                        },
                        "audio": False
                    },
                    async_processing=True,
                )
                
                # Update metrics in real-time
                if ctx.video_processor:
                    try:
                        with ctx.video_processor.lock:
                            rep_metric.metric("💪 Reps", ctx.video_processor.rep_count)
                            step_metric.metric("📍 Step", 
                                f"{ctx.video_processor.current_step + 1}/{exercise_data['num_steps']}")
                            
                            acc = ctx.video_processor.step_accuracy
                            acc_delta = "Good form" if acc >= 70 else "Adjust position"
                            accuracy_metric.metric("🎯 Accuracy", f"{acc:.0f}%", delta=acc_delta)
                    except:
                        rep_metric.metric("💪 Reps", 0)
                        step_metric.metric("📍 Step", f"1/{exercise_data['num_steps']}")
                        accuracy_metric.metric("🎯 Accuracy", "0%")
                
                # Step reference images
                st.markdown("---")
                st.markdown("### 📋 Exercise Steps Reference")
                
                step_cols = st.columns(exercise_data['num_steps'])
                
                for i in range(exercise_data['num_steps']):
                    with step_cols[i]:
                        # Highlight current step
                        if ctx.video_processor:
                            try:
                                current = ctx.video_processor.current_step
                                if i == current:
                                    st.markdown(f"**Step {i+1}** 🔵")
                                elif i < current:
                                    st.markdown(f"**Step {i+1}** ✅")
                                else:
                                    st.markdown(f"**Step {i+1}**")
                            except:
                                st.markdown(f"**Step {i+1}**")
                        else:
                            st.markdown(f"**Step {i+1}**")
                        
                        # Display step image
                        step_paths = exercise_data.get('step_images', [])
                        if i < len(step_paths) and os.path.exists(step_paths[i]):
                            st.image(step_paths[i], use_column_width=True)
                        else:
                            st.warning("No image")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #6b7280;'>PhysioCheck - AI Physiotherapy Assistant</p>", unsafe_allow_html=True)
