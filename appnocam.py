"""
BIRD NEST ANALYSIS - NO CAMERA MODE
Dùng khi camera không hoạt động - Tạo dummy video stream
"""
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from ultralytics import YOLO
import time
import json
import os
from datetime import datetime
from collections import defaultdict
import uuid
import threading
import logging

# Copy toàn bộ code từ app.py nhưng sửa camera detection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'bird-nest-ai-2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ============================================
# CONFIGURATION
# ============================================
MODEL_PATH = 'best.pt'
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
SESSIONS_DIR = 'sessions'
REPORTS_DIR = 'reports'

os.makedirs(SESSIONS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ============================================
# GLOBAL STATE
# ============================================
class SystemState:
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    PROCESSING = "PROCESSING"
    RESULT_READY = "RESULT_READY"
    ERROR = "ERROR"

current_state = SystemState.IDLE
current_session = None
model = None
camera = None
available_cameras = []

class Session:
    def __init__(self, session_id, camera_id):
        self.id = session_id
        self.camera_id = camera_id
        self.start_time = datetime.now()
        self.end_time = None
        self.duration = 0
        self.detections = []
        self.nests = {}
        self.results = {}
        self.state = SystemState.IDLE
    
    def to_dict(self):
        return {
            'id': self.id,
            'camera_id': self.camera_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration,
            'state': self.state,
            'total_detections': len(self.detections),
            'total_nests': len(self.nests),
            'results': self.results
        }
    
    def save(self):
        filepath = os.path.join(SESSIONS_DIR, f"{self.id}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

# ============================================
# LOAD MODEL
# ============================================
def load_yolo_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading YOLO model: {MODEL_PATH}")
            model = YOLO(MODEL_PATH)
            logger.info(f"✅ Model loaded!")
            return True
        else:
            logger.warning(f"⚠️ Model not found: {MODEL_PATH}")
            return False
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return False

# ============================================
# DUMMY CAMERA - KHÔNG CẦN CAMERA THẬT
# ============================================
def detect_cameras():
    """Tạo dummy camera - bypass camera thật"""
    global available_cameras
    
    logger.info("🧪 NO CAMERA MODE: Using dummy camera")
    
    available_cameras = [
        {
            'id': 0,
            'name': 'Virtual Camera (No hardware needed)',
            'resolution': '640x480',
            'backend': 'Dummy',
            'backend_code': cv2.CAP_ANY
        }
    ]
    
    logger.info("✅ Dummy camera ready")
    return available_cameras

def init_camera(camera_id=0):
    """Tạo dummy camera stream"""
    global camera
    
    logger.info("🧪 Initializing dummy camera...")
    
    # Không cần camera thật, chỉ đánh dấu là ready
    camera = True  # Flag để báo camera "sẵn sàng"
    
    logger.info("✅ Dummy camera initialized")
    return True

# ============================================
# DUMMY VIDEO GENERATOR
# ============================================
frame_counter = 0

def generate_frames():
    """Tạo dummy video frames với text"""
    global frame_counter, current_session
    
    while True:
        # Tạo frame đen với text
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add gradient background
        for i in range(480):
            color = int(30 + (i / 480) * 30)
            frame[i, :] = [color, color, color]
        
        # Text
        cv2.putText(frame, "DEMO MODE - NO CAMERA NEEDED", (80, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.putText(frame, "System is working without hardware camera", (90, 250),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Simulate detection boxes (fake)
        if current_state == SystemState.RUNNING and current_session:
            # Draw fake boxes periodically
            if frame_counter % 30 == 0:  # Every 30 frames
                # Fake nest box
                cv2.rectangle(frame, (150, 100), (250, 200), (0, 255, 0), 2)
                cv2.putText(frame, "nest 0.95", (150, 95),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Fake detection
                detection = {
                    'frame_id': frame_counter,
                    'timestamp': time.time(),
                    'class': 'nest',
                    'bbox': [150, 100, 250, 200],
                    'confidence': 0.95
                }
                current_session.detections.append(detection)
                
                socketio.emit('detection', {
                    'class': 'nest',
                    'confidence': 0.95,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Status
            duration = int(time.time() - current_session.start_time.timestamp())
            mins, secs = divmod(duration, 60)
            cv2.putText(frame, f"RECORDING {mins:02d}:{secs:02d}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Detections: {len(current_session.detections)}", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            frame_counter += 1
        else:
            cv2.putText(frame, "Press START to begin analysis", (150, 300),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS

# Copy tất cả functions xử lý từ app.py
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    inter_area = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def process_session(session):
    global current_state
    
    try:
        current_state = SystemState.PROCESSING
        socketio.emit('state_change', {'state': current_state})
        
        socketio.emit('processing_step', {'step': 1, 'message': 'Filtering detections...'})
        time.sleep(0.5)
        
        socketio.emit('processing_step', {'step': 2, 'message': 'Identifying unique nests...'})
        
        nest_detections = [d for d in session.detections if d['class'] == 'nest']
        
        nest_clusters = []
        for det in nest_detections:
            bbox = det['bbox']
            found_cluster = False
            
            for cluster in nest_clusters:
                ref_bbox = cluster[0]['bbox']
                if calculate_iou(bbox, ref_bbox) > IOU_THRESHOLD:
                    cluster.append(det)
                    found_cluster = True
                    break
            
            if not found_cluster:
                nest_clusters.append([det])
        
        logger.info(f"Found {len(nest_clusters)} unique nests")
        time.sleep(0.5)
        
        socketio.emit('processing_step', {'step': 3, 'message': 'Classifying nest status...'})
        
        for idx, cluster in enumerate(nest_clusters):
            nest_id = f"NEST-{idx+1:03d}"
            
            avg_bbox = [
                np.mean([d['bbox'][0] for d in cluster]),
                np.mean([d['bbox'][1] for d in cluster]),
                np.mean([d['bbox'][2] for d in cluster]),
                np.mean([d['bbox'][3] for d in cluster])
            ]
            
            has_egg = False
            has_chick = False
            has_adult = False
            
            for det in session.detections:
                if det['class'] in ['egg', 'chick', 'adult_bird']:
                    if calculate_iou(avg_bbox, det['bbox']) > 0.3:
                        if det['class'] == 'egg':
                            has_egg = True
                        elif det['class'] == 'chick':
                            has_chick = True
                        elif det['class'] == 'adult_bird':
                            has_adult = True
            
            if not has_egg and not has_chick and not has_adult:
                status = "ready"
                reason = "No egg/chick/adult detected"
            else:
                status = "not_ready"
                reasons = []
                if has_egg:
                    reasons.append("egg")
                if has_chick:
                    reasons.append("chick")
                if has_adult:
                    reasons.append("adult")
                reason = f"Contains: {', '.join(reasons)}"
            
            session.nests[nest_id] = {
                'id': nest_id,
                'bbox': avg_bbox,
                'status': status,
                'reason': reason,
                'has_egg': has_egg,
                'has_chick': has_chick,
                'has_adult': has_adult,
                'detections_count': len(cluster),
                'avg_confidence': np.mean([d['confidence'] for d in cluster])
            }
        
        time.sleep(0.5)
        
        socketio.emit('processing_step', {'step': 4, 'message': 'Generating report...'})
        
        total_nests = len(session.nests)
        ready_nests = sum(1 for n in session.nests.values() if n['status'] == 'ready')
        not_ready_nests = total_nests - ready_nests
        ready_rate = (ready_nests / total_nests * 100) if total_nests > 0 else 0
        
        breakdown = {
            'nest_ready': ready_nests,
            'nest_chick': sum(1 for n in session.nests.values() if n['has_chick'] and not n['has_egg'] and not n['has_adult']),
            'nest_adult': sum(1 for n in session.nests.values() if n['has_adult'] and not n['has_egg'] and not n['has_chick']),
            'nest_egg': sum(1 for n in session.nests.values() if n['has_egg'] and not n['has_chick'] and not n['has_adult']),
            'nest_chick_egg': sum(1 for n in session.nests.values() if n['has_chick'] and n['has_egg'] and not n['has_adult']),
            'nest_egg_adult': sum(1 for n in session.nests.values() if n['has_egg'] and n['has_adult'] and not n['has_chick']),
            'nest_chick_adult': sum(1 for n in session.nests.values() if n['has_chick'] and n['has_adult'] and not n['has_egg']),
        }
        
        session.results = {
            'total_nests': total_nests,
            'ready_nests': ready_nests,
            'not_ready_nests': not_ready_nests,
            'ready_rate': round(ready_rate, 2),
            'breakdown': breakdown,
            'total_detections': len(session.detections)
        }
        
        session.save()
        
        current_state = SystemState.RESULT_READY
        socketio.emit('state_change', {'state': current_state})
        socketio.emit('processing_complete', {'results': session.results})
        
        logger.info(f"✅ Processing complete")
        
    except Exception as e:
        logger.error(f"❌ Processing error: {e}")
        current_state = SystemState.ERROR
        socketio.emit('state_change', {'state': current_state, 'error': str(e)})

# ============================================
# ROUTES
# ============================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/system/status')
def system_status():
    return jsonify({
        'state': current_state,
        'model_loaded': model is not None,
        'camera_connected': True,  # Always true in no-camera mode
        'cameras': available_cameras,
        'session': current_session.to_dict() if current_session else None
    })

@app.route('/api/cameras')
def get_cameras():
    return jsonify({'cameras': available_cameras})

@app.route('/api/sessions/history')
def get_history():
    sessions = []
    for filename in os.listdir(SESSIONS_DIR):
        if filename.endswith('.json'):
            with open(os.path.join(SESSIONS_DIR, filename), 'r') as f:
                sessions.append(json.load(f))
    
    sessions.sort(key=lambda x: x['start_time'], reverse=True)
    return jsonify({'sessions': sessions})

@app.route('/api/sessions/<session_id>')
def get_session(session_id):
    filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'Session not found'}), 404

# ============================================
# SOCKET HANDLERS
# ============================================
@socketio.on('start_session')
def handle_start_session(data):
    global current_state, current_session, frame_counter
    
    camera_id = data.get('camera_id', 0)
    
    session_id = str(uuid.uuid4())[:8]
    current_session = Session(session_id, camera_id)
    current_session.state = SystemState.RUNNING
    
    current_state = SystemState.RUNNING
    frame_counter = 0
    
    logger.info(f"✅ Session started: {session_id}")
    emit('session_started', {'session_id': session_id})
    emit('state_change', {'state': current_state})

@socketio.on('stop_session')
def handle_stop_session():
    global current_state, current_session
    
    if current_session:
        current_session.end_time = datetime.now()
        current_session.duration = int((current_session.end_time - current_session.start_time).total_seconds())
        current_session.state = SystemState.STOPPING
        
        current_state = SystemState.STOPPING
        emit('state_change', {'state': current_state})
        
        logger.info(f"⏹️ Session stopped: {current_session.id}")
        
        threading.Thread(target=process_session, args=(current_session,), daemon=True).start()

@socketio.on('get_results')
def handle_get_results():
    if current_session and current_state == SystemState.RESULT_READY:
        emit('results', {
            'session': current_session.to_dict(),
            'nests': list(current_session.nests.values())
        })

# ============================================
# MAIN
# ============================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🪺 BIRD NEST ANALYSIS - NO CAMERA MODE")
    print("=" * 60)
    print("🧪 This version works WITHOUT a physical camera")
    print("📹 Using dummy video stream for testing")
    print("=" * 60)
    
    model_ok = load_yolo_model()
    print(f"Model: {'✅ Loaded' if model_ok else '⚠️ Not found (OK for demo)'}")
    
    detect_cameras()
    print(f"Cameras: ✅ Dummy camera ready")
    
    print("\n🌐 Starting server on http://localhost:5000")
    print("=" * 60 + "\n")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)