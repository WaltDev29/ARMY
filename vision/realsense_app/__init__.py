from typing import List
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from .camera import stop_camera, get_intrinsics, generate_frames, get_detection_data, set_current_targets, get_current_targets
from .debug import start_debug_stream
from .schemas import ResponseBase, Intrinsics
from .convert_pos import get_objects_world_pos

class TargetRequest(BaseModel):
    targets: List[str]


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
        import os
        html_path = os.path.join(os.path.dirname(__file__), "index.html")
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)

    @app.post("/targets", summary="Set Multiple Targets", description="탐지할 대상 목록을 문자열 배열 형태로 지정합니다.")
    def set_targets(request: TargetRequest):
        set_current_targets(request.targets)
        return {"status": "success", "targets": request.targets}

    @app.get("/targets", summary="Get Current Targets", description="현재 지정된 대상 목록을 반환합니다.")
    def get_targets():
        return {"targets": get_current_targets()}



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