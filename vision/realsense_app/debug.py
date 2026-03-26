import cv2
import numpy as np
import threading
from .camera import get_aligned_frames, init_camera

debug_thread = None
is_debug_running = False

def _debug_stream_loop():
    global is_debug_running
    # 지연 임포트를 통해 초기 카메라 구동 속도 저하 방지 및 순환 참조 방지
    from .yolo_detect import detect_objects
    
    while is_debug_running:
        try:
            aligned_frames = get_aligned_frames()
            
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
                y2 = int(cx + w / 2)
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
    global is_debug_running, debug_thread
    
    init_camera()
        
    if not is_debug_running:
        is_debug_running = True
        debug_thread = threading.Thread(target=_debug_stream_loop, daemon=True)
        return debug_thread

def stop_debug_stream():
    global is_debug_running, debug_thread
    is_debug_running = False
    if debug_thread is not None:
        debug_thread.join()
        debug_thread = None
