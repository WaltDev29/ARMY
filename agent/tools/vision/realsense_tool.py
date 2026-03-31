from langchain.tools import tool
from ...core.config import config
import requests
from typing import Any, Dict

def get_realsense_detections() -> Dict[str, Any]:
    """
    RealSense 카메라와 YOLO를 사용하여 현재 프레임의 객체 탐지 결과와 월드 좌표 정보를 반환합니다.
    """
    try:
        resp = requests.get(f"{config.VISION_URL}/detect_world_pos", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "error": str(e), "detections": []}

realsense_detect_tool = tool(
    get_realsense_detections,
    description="""
    RealSense 카메라를 통해 현재 시야에 있는 객체들을 탐지하고, 각 객체의 이름(class_name), 월드 좌표(world_x, world_y, world_z) 정보를 반환합니다.
    반환값 형식 예시:
    {
        "status": "success",
        "detections": [
            {
                "class_name": "person",
                "world_x": 0.1,
                "world_y": 0.2,
                "world_z": 0.3,
            }
        ]
    }
    """
)

tools = [
    realsense_detect_tool
]
