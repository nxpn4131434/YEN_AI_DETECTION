# 🪺 Bird Nest Analysis System

## Hệ Thống Phân Tích Tổ Yến Bằng AI - Enterprise Grade

### 📋 Tổng Quan

Hệ thống phân tích tổ yến tự động sử dụng YOLO để phát hiện và phân loại:
- **nest**: Tổ yến
- **egg**: Trứng
- **chick**: Chim non
- **adult_bird**: Chim trưởng thành

**Mục tiêu**: Xác định tổ nào sẵn sàng thu hoạch (chỉ có tổ, không có egg/chick/adult)

---

## ✨ Tính Năng

### 🎯 Core Features
- ✅ **Session-based Analysis**: Mỗi phiên phân tích độc lập
- ✅ **Realtime Detection**: YOLO detection với bounding boxes
- ✅ **State Machine**: IDLE → RUNNING → PROCESSING → RESULT_READY
- ✅ **Smart Classification**: Tự động phân loại 8 trạng thái tổ
- ✅ **KPI Dashboard**: Thống kê tổng quan & biểu đồ
- ✅ **Session History**: Lưu trữ & xem lại kết quả
- ✅ **Export Report**: Xuất báo cáo chi tiết

### 📊 Processing Pipeline
1. **Capture**: Thu thập frames từ camera
2. **Inference**: YOLO detect objects realtime
3. **Deduplication**: Gom nhóm tổ duy nhất (IoU clustering)
4. **Classification**: Phân loại dựa trên rule-based logic
5. **Aggregation**: Tính toán KPIs & statistics
6. **Report**: Tạo báo cáo JSON + visualization

---

## 🏗️ Kiến Trúc

### Backend (Flask + SocketIO)
```
app.py
├── YOLO Model Loading
├── Camera Management (multi-camera support)
├── Session Management
├── Detection Pipeline
├── Processing & Aggregation
├── REST API Endpoints
└── WebSocket Events
```

### Frontend (HTML + JS + Chart.js)
```
6 Screens:
├── A. Home/Landing (System status + Start)
├── B. Session Running (Realtime + Controls)
├── C. Processing (Progress indicator)
├── D. Results Dashboard (KPIs + Charts + Table)
├── E. Nest Detail (Chi tiết từng tổ)
└── F. Session History (Lịch sử phiên)
```

---

## 🚀 Cài Đặt

### Bước 1: Clone & Dependencies
```bash
cd bird_nest_system
pip install -r requirements.txt
```

### Bước 2: Chuẩn Bị Model
Đặt file YOLO model `best.pt` vào thư mục gốc.

**Model phải có 4 classes**:
- `nest`
- `egg`
- `chick`
- `adult_bird`

### Bước 3: Kiểm Tra Camera
```bash
python -c "import cv2; print('Camera 0:', cv2.VideoCapture(0).isOpened())"
```

### Bước 4: Chạy Server
```bash
python app.py
```

Server sẽ chạy tại: **http://localhost:5000**

---

## 📐 Cấu Trúc Thư Mục

```
bird_nest_system/
├── app.py                    # Backend Flask
├── templates/
│   └── index.html           # Frontend (6 screens)
├── sessions/                # Session data (auto-created)
│   └── *.json
├── reports/                 # Reports (auto-created)
│   └── *.json
├── best.pt                  # YOLO model (bạn cần thêm)
├── requirements.txt
└── README.md
```

---

## 🎮 Hướng Dẫn Sử Dụng

### 1️⃣ Screen A: Home
1. Kiểm tra **Camera Status** = Connected
2. Kiểm tra **Model Status** = Ready
3. Chọn camera từ dropdown
4. Nhấn **"BẮT ĐẦU PHÂN TÍCH"**

### 2️⃣ Screen B: Session Running
- Quan sát video stream với bounding boxes
- Xem detections realtime (panel bên phải)
- Theo dõi timer & số lượng detections
- Nhấn **"KẾT THÚC & TỔNG HỢP"** khi xong

### 3️⃣ Screen C: Processing
- Hệ thống tự động xử lý 4 bước:
  1. Filter detections
  2. Deduplicate nests
  3. Classify status
  4. Generate report
- Đợi progress bar = 100%

### 4️⃣ Screen D: Results Dashboard
- Xem **KPIs**: Tổng tổ, Ready, Not ready, Tỷ lệ %
- **Breakdown**: Phân loại 8 trạng thái
- **Biểu đồ**: Bar chart phân bố
- **Bảng chi tiết**: Danh sách tất cả tổ
- **Xuất báo cáo**: Download JSON/CSV

### 5️⃣ Screen F: History
- Xem tất cả phiên đã chạy
- Filter theo ngày, camera
- Click vào session để xem lại kết quả

---

## 🧠 Logic Phân Loại

### Rule-Based Classification

| Trạng Thái | Điều Kiện | Harvest? |
|-----------|-----------|----------|
| **nest_only** | Chỉ có nest, không có egg/chick/adult | ✅ **Yes** |
| **nest_egg** | Nest + Egg | ❌ No |
| **nest_chick** | Nest + Chick | ❌ No |
| **nest_adult** | Nest + Adult | ❌ No |
| **nest_egg_chick** | Nest + Egg + Chick | ❌ No |
| **nest_egg_adult** | Nest + Egg + Adult | ❌ No |
| **nest_chick_adult** | Nest + Chick + Adult | ❌ No |
| **all_present** | Nest + Egg + Chick + Adult | ❌ No |

### Thuật Toán Deduplicate

1. Lọc tất cả detections có class = `nest`
2. Cluster theo IoU (Intersection over Union)
3. Nếu IoU > 0.45 → Cùng 1 nest
4. Mỗi cluster = 1 unique nest

### Thuật Toán Gán Trạng Thái

```python
for each nest:
    avg_bbox = average(cluster_bboxes)
    
    has_egg = any(egg_bbox overlap avg_bbox > 0.3)
    has_chick = any(chick_bbox overlap avg_bbox > 0.3)
    has_adult = any(adult_bbox overlap avg_bbox > 0.3)
    
    if not (has_egg or has_chick or has_adult):
        status = "ready"  # ✅ Sẵn sàng thu hoạch
    else:
        status = "not_ready"  # ❌ Không thu hoạch
```

---

## ⚙️ Configuration

### Backend Settings (app.py)

```python
# Line 32-36
MODEL_PATH = 'best.pt'              # YOLO model path
CONFIDENCE_THRESHOLD = 0.35         # Detection confidence
IOU_THRESHOLD = 0.45                # Nest clustering threshold
SESSIONS_DIR = 'sessions'           # Session storage
REPORTS_DIR = 'reports'             # Report storage
```

### Thay Đổi Thông Số

| Parameter | Mô Tả | Giá Trị Đề Xuất |
|-----------|-------|-----------------|
| `CONFIDENCE_THRESHOLD` | Ngưỡng confidence tối thiểu | 0.3 - 0.5 |
| `IOU_THRESHOLD` | Ngưỡng gom nhóm tổ | 0.4 - 0.6 |
| Overlap threshold (line 384) | Gán egg/chick/adult vào nest | 0.2 - 0.4 |

---

## 📡 API Endpoints

### REST API

```
GET  /                          → Trang chủ
GET  /video_feed                → Video stream
GET  /api/system/status         → System status
GET  /api/cameras               → Danh sách camera
GET  /api/sessions/history      → Lịch sử sessions
GET  /api/sessions/<id>         → Chi tiết 1 session
```

### WebSocket Events

#### Client → Server
```javascript
socket.emit('start_session', {camera_id: 0})
socket.emit('stop_session')
socket.emit('get_results')
```

#### Server → Client
```javascript
socket.on('session_started', {session_id})
socket.on('state_change', {state})
socket.on('detection', {class, confidence, timestamp})
socket.on('processing_step', {step, message})
socket.on('processing_complete', {results})
socket.on('results', {session, nests})
```

---

## 🎨 UI/UX Design

### Theme
- **Colors**: Natural green (#2d5016) + brown (#8b6914)
- **Font**: Inter (modern, clean)
- **Style**: Enterprise-grade, professional
- **Animations**: Subtle transitions & loading states

### Responsive
- Desktop: Optimized for 1280x720+
- Tablet: Adaptive grid layout
- Mobile: Stack vertical layout

---

## 🐛 Troubleshooting

### ❌ "Model not found"
```bash
# Đảm bảo best.pt nằm cùng thư mục app.py
ls -la best.pt
```

### ❌ "No camera detected"
```bash
# Kiểm tra camera khả dụng
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened()); cap.release()"

# Thử camera khác
python -c "import cv2; cap = cv2.VideoCapture(1); print(cap.isOpened()); cap.release()"
```

### ❌ "Processing fails"
- Kiểm tra có ít nhất 1 nest detection
- Kiểm tra confidence threshold (có thể giảm xuống 0.25)
- Xem logs trong terminal

### ❌ "Browser không mở"
```bash
# Mở thủ công
http://localhost:5000
```

---

## 📊 KPI Metrics

### Session Metrics
- **Total Nests**: Tổng số tổ duy nhất
- **Ready Nests**: Số tổ sẵn sàng thu hoạch
- **Not Ready Nests**: Số tổ không thu hoạch
- **Ready Rate**: % tổ sẵn sàng
- **Total Detections**: Tổng lượng detections
- **Duration**: Thời gian phiên (seconds)

### Per-Nest Metrics
- **Nest ID**: Định danh duy nhất
- **Status**: ready / not_ready
- **Reason**: Lý do (có egg/chick/adult)
- **Confidence**: Trung bình confidence
- **Detections Count**: Số lần xuất hiện

---

## 🔮 Roadmap

### Phase 1 (Current)
- ✅ Basic detection & classification
- ✅ Session management
- ✅ Dashboard & reports

### Phase 2 (Next)
- ⏳ Video file upload support
- ⏳ Advanced tracking (ByteTrack/DeepSORT)
- ⏳ Export to CSV/Excel
- ⏳ Nest location mapping (grid zones)

### Phase 3 (Future)
- ⏳ Multi-user support
- ⏳ Cloud storage integration
- ⏳ Mobile app
- ⏳ Alert notifications

---

## 🤝 Contributing

Đóng góp code:
1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

---

## 📄 License

MIT License - Free to use for commercial projects

---

## 📞 Support

- 📧 Email: support@birdnest.ai
- 📖 Docs: https://docs.birdnest.ai
- 💬 Community: https://forum.birdnest.ai

---

## 🎉 Kết Luận

Hệ thống **Bird Nest Analysis** cung cấp giải pháp hoàn chỉnh, enterprise-grade cho việc phân tích tổ yến tự động. 

**Key Highlights**:
- 🎯 Chính xác cao với YOLO
- 🚀 Xử lý realtime
- 📊 Dashboard trực quan
- 🔄 Session management chuyên nghiệp
- 📈 Scalable & maintainable

**Sẵn sàng triển khai ngay!** 🪺✨
