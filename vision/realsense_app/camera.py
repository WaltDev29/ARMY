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



