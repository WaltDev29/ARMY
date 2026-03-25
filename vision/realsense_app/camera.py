import pyrealsense2 as rs
import numpy as np
import cv2
import threading

# 전역 변수 설정
pipeline = None
config = None
align = None
is_streaming = False
debug_thread = None
is_debug_running = False

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

def get_rgb_image():
    """카메라를 통해 2D RGB 이미지를 반환하는 함수"""
    global pipeline, align, is_streaming
    if not is_streaming:
        init_camera()
        
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    
    if not color_frame:
        return None
        
    return np.asanyarray(color_frame.get_data())

def get_depth_data():
    """카메라를 통해 Depth 데이터를 반환하는 함수"""
    global pipeline, align, is_streaming
    if not is_streaming:
        init_camera()
        
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    
    if not depth_frame:
        return None
        
    # 밀리미터 단위의 depth 데이터 배열 (640x480)
    return np.asanyarray(depth_frame.get_data())

def _debug_stream_loop():
    global is_debug_running, pipeline, align
    # 지연 임포트를 통해 초기 카메라 구동 속도 저하 방지 및 순환 참조 방지
    from .yolo_detect import detect_objects
    
    while is_debug_running:
        try:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            aligned_frames = align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
                
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # YOLO 추론을 수행하고 RGB 이미지에 바운딩 박스를 그립니다.
            detections = detect_objects(color_image)
            for obj in detections:
                cx, cy, w, h = obj["xywh"]
                class_name = obj["class_name"]
                
                # 중심 좌표 및 너비/높이 속성을 좌상단, 우하단 좌표로 변환
                x1 = int(cx - w / 2)
                y1 = int(cy - h / 2)
                x2 = int(cx + w / 2)
                y2 = int(cy + h / 2)
                
                # 바운딩 박스와 클래스명 표시 (RGB 이미지에만)
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(color_image, class_name, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Depth 이미지를 시각화하기 위해 컬러맵 적용
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            # RGB 이미지와 Depth 이미지를 가로로 연결
            images = np.hstack((color_image, depth_colormap))
            
            cv2.imshow('RealSense Debug Stream', images)
        except Exception as e:
            print(f"Debug stream error: {e}")
            break
            
        key = cv2.waitKey(1)
        # 'q' 키를 누르면 종료
        if key & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()
    is_debug_running = False

def start_debug_stream():
    """디버깅을 위해 카메라 프레임을 화면에 표시하는 Threading 함수"""
    global is_debug_running, debug_thread, is_streaming
    
    if not is_streaming:
        init_camera()
        
    if not is_debug_running:
        is_debug_running = True
        t = threading.Thread(target=_debug_stream_loop, daemon=True)
        return t

def stop_camera():
    global is_streaming, is_debug_running, debug_thread, pipeline
    is_debug_running = False
    if debug_thread is not None:
        debug_thread.join()
        
    if is_streaming and pipeline is not None:
        pipeline.stop()
        is_streaming = False

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
    
