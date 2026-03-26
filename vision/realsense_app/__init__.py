from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from .camera import stop_camera, get_intrinsics, generate_frames, get_detection_data
from .debug import start_debug_stream
from .schemas import ResponseBase, Intrinsics
from .convert_pos import get_objects_world_pos

import os
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR/".env")

DEBUG = os.getenv("VISION_DEBUG", "false").lower() == "true"



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
            return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')



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
        data = get_detection_data()
        if isinstance(data, dict) and "error" in data:
            return data
            
        return ResponseBase(
            status="success",
            detections=data
        )

    
    @app.get("/detect_world_pos",
             summary="Detect-World-Pos",
             description="오브젝트들의 월드 좌표를 반환합니다.",
             response_model=ResponseBase)
    async def detect_world_pos():
        data = get_objects_world_pos()
        if isinstance(data, dict) and "error" in data:
            return data
            
        return ResponseBase(
            status="success",
            detections=data
        )

    
    @app.get("/cam_state",
             summary="Camera State",
             description="카메라의 고유 스펙 (intrinsics)과 현재 pitch, roll 각도를 반환합니다.",
             response_model=Intrinsics)
    def get_camera_state():
        return Intrinsics(**get_intrinsics())
        

    return app