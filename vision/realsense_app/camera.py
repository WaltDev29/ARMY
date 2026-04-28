import pyrealsense2 as rs
import numpy as np
import cv2
import threading

# 전역 변수 설정
pipeline = None
config = None
align = None
is_streaming = False
CURRENT_TARGETS = []

def set_current_targets(targets: list):
    global CURRENT_TARGETS
    CURRENT_TARGETS = targets

def get_current_targets() -> list:
    return CURRENT_TARGETS

# ============ Init Camera ============
def init_camera():
    global pipeline, config, align, is_streaming
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
    
    is_streaming = True



# ============ Fetch Aligned Frames ============
def get_aligned_frames():
    global pipeline, align, is_streaming
    if not is_streaming:
        init_camera()
        
    frames = pipeline.wait_for_frames()
    return align.process(frames)



# ============ Get RGB Image ============
def get_rgb_image():
    """카메라를 통해 2D RGB 이미지를 반환하는 함수"""
    aligned_frames = get_aligned_frames()
    color_frame = aligned_frames.get_color_frame()
    
    if not color_frame:
        return None
        
    return np.asanyarray(color_frame.get_data())



# ============ Get Depth Data ============
def get_depth_data():
    """카메라를 통해 Depth 데이터를 반환하는 함수"""
    aligned_frames = get_aligned_frames()
    depth_frame = aligned_frames.get_depth_frame()
    
    if not depth_frame:
        return None
        
    # 밀리미터 단위의 depth 데이터 배열 (640x480)
    return np.asanyarray(depth_frame.get_data())



# ============ Stop Camera ============
def stop_camera():
    global is_streaming, pipeline
    from .debug import stop_debug_stream
    stop_debug_stream()
        
    if is_streaming and pipeline is not None:
        pipeline.stop()
        is_streaming = False




# ============ Camera 고유 스펙 + 각도 반환 ============
def get_intrinsics():
    """RealSense 카메라의 고유 스펙(Intrinsics)을 반환합니다."""
    global pipeline, is_streaming
    if not is_streaming:
        init_camera()
    
    try:
        profile = pipeline.get_active_profile()
        color_stream = profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        # ====== 각도 계산 ======
        frames = pipeline.wait_for_frames()
        accel = frames.first_or_default(rs.stream.accel)

        pitch = 0.0
        roll = 0.0

        if accel:
            data = accel.as_motion_frame().get_motion_data()

            ax, ay, az = data.x, data.y, data.z

            pitch = np.arctan2(-ax, np.sqrt(ay*ay + az*az)) * 180/np.pi
            roll  = np.arctan2(ay, az) * 180/np.pi


        return {
            "width": intrinsics.width,
            "height": intrinsics.height,
            "ppx": intrinsics.ppx,
            "ppy": intrinsics.ppy,
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "pitch": pitch,
            "roll" : roll
        }
    except Exception as e:
        print(f"Error getting intrinsics: {e}")
        return None




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
                continue

            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
        except Exception as e:
            print(f"Stream error: {e}")
            break




# ============ depth 데이터 & YOLO 추론 결과 반환 ============
def get_detection_data():
    """카메라에서 이미지를 받아 depth 데이터와 YOLO 추론 결과를 반환합니다."""
    from .yolo_detect import detect_objects
    
    # 1. 카메라를 통해 2D RGB 이미지 및 Depth 데이터 가져오기
    aligned_frames = get_aligned_frames()
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not color_frame or not depth_frame:
        return {"error": "카메라에서 프레임을 가져오지 못했습니다."}

    rgb_image = np.asanyarray(color_frame.get_data())
    depth_data = np.asanyarray(depth_frame.get_data())

    # 2. YOLO를 통해 오브젝트 탐지
    detected_objects = detect_objects(rgb_image)

    results = []
    # depth_data의 shape는 (height, width) 형태
    height, width = depth_data.shape
    
    for obj in detected_objects:
        x, y, w, h = obj["xywh"]
        # 바운딩 박스 중심점 (cx, cy)
        cx, cy = int(x), int(y)
        
        # 깊이 데이터가 이미지 해상도 범위를 벗어나지 않도록 예외 처리
        if 0 <= cy < height and 0 <= cx < width:
            # depth_data의 단위는 밀리미터(mm)
            depth_value = float(depth_data[cy, cx])
        else:
            depth_value = 0.0
            
        results.append({
            "class_name": obj["class_name"],
            "xywh": [x, y, w, h],
            "distance_mm": depth_value
        })

    return results
