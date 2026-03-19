from langchain.tools import tool
from ...core.config import config
import requests
from typing import Any, Dict

def get_realsense_detections() -> Dict[str, Any]:
    """
    RealSense 카메라와 YOLO를 사용하여 현재 프레임의 객체 탐지 결과와 거리(Depth) 정보를 반환합니다.
    """
    try:
        resp = requests.get(f"{config.VISION_URL}/detect", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "error": str(e), "detections": []}

realsense_detect_tool = tool(
    get_realsense_detections,
    description="""
    RealSense 카메라를 통해 현재 시야에 있는 객체들을 탐지하고, 각 객체의 이름(class), 화면상 위치(xywh), 그리고 카메라부터의 거리(distance_mm) 정보를 반환합니다.
    주변 환경의 객체 인식과 깊이(거리) 파악이 필요할 때 이 도구를 사용하세요.
    
    반환값 형식 예시:
    {
        "status": "success",
        "detections": [
            {
                "class_name": "person",
                "xywh": [320, 240, 100, 200],  # 중심x, 중심y, 너비, 높이
                "distance_mm": 1500.0          # 카메라로부터의 거리 (밀리미터 단위)
            }
        ]
    }
    """
)

tools = [
    realsense_detect_tool
]
