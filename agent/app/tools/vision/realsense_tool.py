from langchain.tools import tool
from ...core.config import config
import requests
from typing import Any, Dict

def get_realsense_detections(targets: list[str] = None) -> Dict[str, Any]:
    """
    RealSense 카메라와 YOLO를 사용하여 현재 프레임의 객체 탐지 결과와 월드 좌표 정보를 반환합니다.
    """
    if targets is None:
        targets = []
        
    try:
        resp = requests.post(f"{config.VISION_URL}/detect_world_pos", json={"targets": targets}, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "error": str(e), "detections": []}

def reset_vision_targets() -> Dict[str, Any]:
    """
    Vision 모듈의 전역 타겟 설정을 초기화(빈 목록)하여, 화면에 보이는 모든 기본 객체를 다시 탐지하도록 만듭니다.
    """
    try:
        resp = requests.post(f"{config.VISION_URL}/targets", json={"targets": []}, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "error": str(e)}

reset_targets_tool = tool(
    reset_vision_targets,
    description="""
    Vision 모듈의 전역 타겟 설정을 초기화(빈 목록)합니다.
    이전에 특정 물체(예: 'apple')만 찾도록 설정된 필터링을 해제하고, 다시 모든 객체를 탐지하는 기본 상태로 되돌립니다.
    """
)

realsense_detect_tool = tool(
    get_realsense_detections,
    description="""
    RealSense 카메라를 통해 현재 시야에 있는 객체들을 탐지하고, 각 객체의 이름(class_name), 월드 좌표(world_x, world_y, world_z) 정보를 반환합니다.
    - targets: 찾고자 하는 특정 객체의 이름 리스트 (예: ["apple", "red block"]).
               **중요**: 사용자가 대화 중에 특정 물체(예: '사과', '블록')를 찾거나 조작하길 명시하는 경우, 반드시 그 물체를 영어로 번역하여 이 targets 파라미터에 명시해야 합니다. (예: "사과 찾아줘" -> ["apple"])
               생략하거나 빈 리스트를 전달하면 이전에 설정된 전역 타겟 설정을 무시하고 현재 화면의 모든 기본 객체를 반환합니다.
    
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
    realsense_detect_tool,
    reset_targets_tool
]
