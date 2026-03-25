from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from .camera import get_rgb_image, get_depth_data, start_debug_stream, stop_camera, get_intrinsics
from .yolo_detect import detect_objects
from .schemas import ResponseBase, Intrinsics

import os
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR/".env")

DEBUG = os.getenv("VISION_DEBUG").lower() == "true"



def create_app() -> FastAPI:
    app = FastAPI()

    @app.on_event("shutdown")
    async def shutdown_event():
        # 앱 종료 시 카메라 리소스를 반환합니다.
        stop_camera()

    @app.get("/")
    def root():
        return "RealSense is running."


    if not DEBUG:
        @app.get("/stream")
        async def stream():
            def frame_generator():
                import cv2

                while True:
                    frame = get_rgb_image()
                    if frame is None:
                        continue

                    # YOLO object detection을 수행하고 바운딩 박스를 그립니다.
                    detections = detect_objects(frame)
                    for obj in detections:
                        x, y, w, h = obj.get("xywh", [0,0,0,0])
                        class_name = obj.get("class_name", "")

                        # xywh는 중심좌표 + w,h이므로, 좌상단 우하단으로 변환
                        x1 = int(x - w / 2)
                        y1 = int(y - h / 2)
                        x2 = int(x + w / 2)
                        y2 = int(y + h / 2)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, class_name, (x1, max(y1 - 5, 0)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    ret, jpeg = cv2.imencode('.jpg', frame)
                    if not ret:
                        continue

                    frame_bytes = jpeg.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            return StreamingResponse(frame_generator(), media_type='multipart/x-mixed-replace; boundary=frame')



    @app.get("/detect",
             summary="Detect",
             description="""
             카메라에서 이미지를 받아 depth 데이터와 YOLO 추론 결과를 반환합니다.
             """,
             response_model=ResponseBase)
    async def detect():
        """
        카메라에서 이미지를 받아 depth 데이터와 YOLO 추론 결과를 반환합니다.
        """
        # 1. 카메라를 통해 2D RGB 이미지 및 Depth 데이터 가져오기
        rgb_image = get_rgb_image()
        depth_data = get_depth_data()

        if rgb_image is None or depth_data is None:
            return {"error": "카메라에서 프레임을 가져오지 못했습니다."}

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

        return ResponseBase(
            status="success",
            detections=results
        )
    
    @app.get("/cam_state",
             summary="Camera State",
             description="카메라의 고유 스펙 (intrinsics)과 현재 pitch, roll 각도를 반환합니다.",
             response_model=Intrinsics)
    def get_camera_state():
        return Intrinsics(**get_intrinsics())
        

    return app