import requests
import json
import time

print("=" * 60)
print("🔍 DEBUG API - Kiểm tra Backend Response")
print("=" * 60)

# Wait for server to start
print("\n⏳ Đợi server khởi động (3 giây)...")
time.sleep(3)

# Check system status
print("\n[1] Checking /api/system/status")
print("-" * 60)

try:
    response = requests.get('http://localhost:5000/api/system/status')
    
    if response.status_code == 200:
        data = response.json()
        
        print("✅ API Response OK\n")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
        print("\n" + "=" * 60)
        print("📊 ANALYSIS")
        print("=" * 60)
        
        # Camera connected
        if data.get('camera_connected'):
            print("✅ camera_connected: TRUE")
            print("   → Web UI sẽ hiển thị 'Connected'")
        else:
            print("❌ camera_connected: FALSE")
            print("   → Web UI sẽ hiển thị 'Disconnected'")
            print("\n🔧 FIX:")
            print("   1. Stop server (Ctrl+C)")
            print("   2. Chạy lại: python app.py")
            print("   3. Xem logs có 'Camera X ready' không")
        
        # Model loaded
        if data.get('model_loaded'):
            print("\n✅ model_loaded: TRUE")
        else:
            print("\n⚠️ model_loaded: FALSE")
            print("   → Cần file best.pt")
        
        # Cameras list
        cameras = data.get('cameras', [])
        print(f"\n📹 Cameras found: {len(cameras)}")
        for cam in cameras:
            print(f"   - {cam['name']}: {cam['resolution']}")
        
    else:
        print(f"❌ API Error: Status {response.status_code}")
        
except requests.exceptions.ConnectionError:
    print("❌ Cannot connect to server")
    print("\n🔧 FIX:")
    print("   1. Đảm bảo server đang chạy")
    print("   2. Chạy trong terminal khác: python app.py")
    print("   3. Đợi thấy 'Starting server on http://localhost:5000'")
    print("   4. Chạy lại script này")

except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 60)

# Check cameras endpoint
print("\n[2] Checking /api/cameras")
print("-" * 60)

try:
    response = requests.get('http://localhost:5000/api/cameras')
    
    if response.status_code == 200:
        data = response.json()
        cameras = data.get('cameras', [])
        
        print(f"✅ Found {len(cameras)} camera(s):\n")
        print(json.dumps(cameras, indent=2, ensure_ascii=False))
    else:
        print(f"❌ Status {response.status_code}")
        
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 60)
print("💡 TIPS")
print("=" * 60)
print("""
Nếu camera_connected = FALSE:
1. Check logs khi start app
2. Phải thấy: "✅ Camera X ready"
3. Nếu không thấy → Camera init failed
4. Chạy: python simple_camera_test.py
5. Hoặc dùng: python app_no_camera.py
""")