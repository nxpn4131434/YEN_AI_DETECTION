import cv2
import sys

print("=" * 60)
print("🔍 CAMERA DEBUG TOOL")
print("=" * 60)

print("\n[1] Kiểm tra OpenCV version:")
print(f"   OpenCV: {cv2.__version__}")

print("\n[2] Thử mở camera với các backend khác nhau:")
print("-" * 60)

backends = [
    (cv2.CAP_DSHOW, "DirectShow (Windows - Tốt nhất)"),
    (cv2.CAP_MSMF, "Media Foundation (Windows)"),
    (cv2.CAP_ANY, "Auto Detect"),
    (cv2.CAP_V4L2, "V4L2 (Linux)"),
]

working_cameras = []

for cam_id in range(5):
    print(f"\n📹 Testing Camera {cam_id}:")
    
    for backend, backend_name in backends:
        try:
            cap = cv2.VideoCapture(cam_id, backend)
            
            if cap.isOpened():
                # Thử đọc frame
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    h, w, _ = frame.shape
                    print(f"   ✅ {backend_name}: {w}x{h}")
                    
                    # Lưu camera hoạt động
                    if cam_id not in [c['id'] for c in working_cameras]:
                        working_cameras.append({
                            'id': cam_id,
                            'backend': backend,
                            'backend_name': backend_name,
                            'resolution': f"{w}x{h}"
                        })
                    
                    cap.release()
                    break  # Tìm thấy rồi, không cần test backend khác
                else:
                    print(f"   ❌ {backend_name}: Opened but can't read frame")
            else:
                print(f"   ❌ {backend_name}: Can't open")
            
            cap.release()
            
        except Exception as e:
            print(f"   ❌ {backend_name}: Error - {e}")

print("\n" + "=" * 60)
print("📊 KẾT QUẢ")
print("=" * 60)

if working_cameras:
    print(f"\n✅ Tìm thấy {len(working_cameras)} camera:")
    for cam in working_cameras:
        print(f"\n   Camera {cam['id']}:")
        print(f"   - Backend: {cam['backend_name']}")
        print(f"   - Resolution: {cam['resolution']}")
        print(f"   - Backend Code: {cam['backend']}")
else:
    print("\n❌ KHÔNG TÌM THẤY CAMERA NÀO!")
    print("\n🔧 HƯỚNG DẪN FIX:")
    print("   1. Kiểm tra camera đã cắm vào chưa")
    print("   2. Kiểm tra Device Manager (Windows):")
    print("      - Win + X → Device Manager")
    print("      - Imaging devices → Tìm camera")
    print("   3. Tắt app khác đang dùng camera:")
    print("      - Zoom, Teams, Skype, OBS...")
    print("      - Task Manager → End Task")
    print("   4. Cài lại driver camera")
    print("   5. Thử khởi động lại máy")

print("\n" + "=" * 60)
print("💡 KHUYẾN NGHỊ CHO app.py")
print("=" * 60)

if working_cameras:
    print("\nThêm code này vào app.py để fix:")
    print("\n```python")
    print("# Line 117-125 trong detect_cameras()")
    print("backends = [")
    for cam in working_cameras[:1]:  # Lấy backend tốt nhất
        print(f"    ({cam['backend']}, '{cam['backend_name']}'),  # ← WORKING!")
    print("    (cv2.CAP_ANY, 'Auto'),")
    print("]")
    print("```")
else:
    print("\n⚠️ Không có camera nào hoạt động!")
    print("   Hệ thống sẽ chạy với dummy camera (No video)")

print("\n" + "=" * 60)