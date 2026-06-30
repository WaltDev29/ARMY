import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import time

# 전역 변수 설정
pipeline = None
config = None
align = None
is_streaming = False
CURRENT_TARGETS = []

# --- 스레드 및 캐시 전역 변수 ---
frame_lock = threading.Lock()
latest_color_frame = None
latest_depth_frame = None
latest_pitch = 0.0
latest_roll = 0.0
cached_intrinsics = None
worker_thread = None
stop_event = threading.Event()

def _frame_worker():
    global pipeline, align, is_streaming, stop_event
    global latest_color_frame, latest_depth_frame, latest_pitch, latest_roll
    
    while not stop_event.is_set():
        if not is_streaming or pipeline is None:
            time.sleep(0.05)
            continue
            
        try:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            aligned_frames = align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            accel = frames.first_or_default(rs.stream.accel)
            
            pitch = 0.0
            roll = 0.0
            if accel:
                data = accel.as_motion_frame().get_motion_data()
                ax, ay, az = data.x, data.y, data.z
                pitch = np.arctan2(-ax, np.sqrt(ay*ay + az*az)) * 180/np.pi
                roll  = np.arctan2(ay, az) * 180/np.pi
            
            if color_frame and depth_frame:
                color_data = np.asanyarray(color_frame.get_data())
                depth_data = np.asanyarray(depth_frame.get_data())
                
                with frame_lock:
                    latest_color_frame = color_data.copy()
                    latest_depth_frame = depth_data.copy()
                    latest_pitch = pitch
                    latest_roll = roll
        except Exception as e:
            time.sleep(0.01)

def set_current_targets(targets: list):
    global CURRENT_TARGETS
    CURRENT_TARGETS = targets

def get_current_targets() -> list:
    return CURRENT_TARGETS

# ============ Init Camera ============
def init_camera():
    global pipeline, config, align, is_streaming, cached_intrinsics, worker_thread, stop_event
    if is_streaming:
        return

    pipeline = rs.pipeline()
    config = rs.config()
    
    # 640x480 해상도, 30fps로 Depth와 Color 스트림 설정
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.accel)
    
    # 설정 적용 및 파이프라인 시작
    pipeline.start(config)
    
    # Depth 프레임을 Color 프레임에 맞추기 위한 Align 객체
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    # 카메라 고유 스펙을 한 번만 가져와 캐싱
    profile = pipeline.get_active_profile()
    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()
    cached_intrinsics = {
        "width": intr.width,
        "height": intr.height,
        "ppx": intr.ppx,
        "ppy": intr.ppy,
        "fx": intr.fx,
        "fy": intr.fy
    }
    
    is_streaming = True
    stop_event.clear()
    
    # 프레임 수신용 백그라운드 스레드 시작
    worker_thread = threading.Thread(target=_frame_worker, daemon=True)
    worker_thread.start()


# ============ Fetch Aligned Frames ============
def get_latest_frames_data():
    """가장 최근에 수신된 컬러(RGB) 및 깊이(Depth) Numpy 배열을 반환합니다."""
    global is_streaming, latest_color_frame, latest_depth_frame
    if not is_streaming:
        init_camera()
        
    # 프레임이 갱신될 때까지 최대 1초 대기
    for _ in range(50):
        with frame_lock:
            if latest_color_frame is not None and latest_depth_frame is not None:
                return latest_color_frame.copy(), latest_depth_frame.copy()
        time.sleep(0.02)
    return None, None


# ============ Stop Camera ============
def stop_camera():
    global is_streaming, pipeline, stop_event, worker_thread
    from .debug import stop_debug_stream
    stop_debug_stream()
        
    if is_streaming:
        stop_event.set()
        if worker_thread is not None:
            worker_thread.join(timeout=1.0)
            
        if pipeline is not None:
            pipeline.stop()
        is_streaming = False


# ============ Camera 고유 스펙 + 각도 반환 ============
def get_intrinsics():
    """RealSense 카메라의 고유 스펙(Intrinsics)과 최신 각도를 반환합니다."""
    global is_streaming, cached_intrinsics, latest_pitch, latest_roll
    if not is_streaming:
        init_camera()
    
    # 캐싱된 값이 없을 때까지 잠시 대기
    for _ in range(50):
        if cached_intrinsics is not None:
            break
        time.sleep(0.02)
        
    if cached_intrinsics is None:
        print("Error getting intrinsics: cached data is None")
        return None

    with frame_lock:
        pitch = latest_pitch
        roll = latest_roll

    intr = cached_intrinsics.copy()
    intr["pitch"] = pitch
    intr["roll"] = roll
    return intr


# ============ 웹 스트리밍을 위한 프레임 제너레이터 ============
def generate_frames():
    """웹 스트리밍을 위한 프레임 제너레이터"""
    from .convert_pos import get_world_coordinates
    import cv2
    
    while True:
        try:
            # 기존 컬러 프레임 로직 대신 convert_pos에 통합된 추론 및 드로잉 결과를 바로 가져옵니다
            world_objects, frame = get_world_coordinates(target_classes=CURRENT_TARGETS, return_image=True)
            
            if frame is None:
                time.sleep(0.01) # CPU 과점유 방지
                continue

            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Stream error: {e}")
            break
