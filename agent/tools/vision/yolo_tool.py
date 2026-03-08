import base64

from langchain.tools import tool
from ...core.config import config
import requests
from pydantic import BaseModel
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR.parent.parent / "memory/images"



# ============ Vision API로 객체 탐지하는 Tool ============ 
def detect_objects():
    response = requests.get(f"{config.VISION_URL}/detect")
    return response.json()

detect_objects_tool = tool(
    detect_objects,
    description="""
    카메라에 보이는 객체들의 정보를 반환합니다. 
    반환값 : [{class:객체명, box:[x1,y1,x2,y2]}, ...]
    반환값 중 box는 객체의 바운딩 박스를 나타내며, x1,y1은 왼쪽 위 꼭지점의 좌표, x2,y2는 오른쪽 아래 꼭지점의 좌표입니다. 
    """
)



# ============ 카메라로 이미지 읽어 저장하는 Tool ============ 
def _get_image() -> cv2.Mat:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    
    if not ret:
        return ""

    cap.release()
    return frame




# ============ Vision API에 이미지를 보내 객체 탐지하는 Tool ============ 
def detect_objects_from_image():
    if config.DOFBOT:
        image = _get_image()
        _, encoded = cv2.imencode('.jpg', image)
        img = encoded.tobytes()
    else :
        from ..robot.pybullet_tool import get_image
        img = get_image()

    if img is None:
        raise ValueError(f"이미지를 불러올 수 없습니다")

    files = {
        'data': ('image.jpg', img, 'image/jpeg')
    }

    response = requests.post(
        config.VISION_URL + "/detect_from_image",
        files=files
    )

    return response.json()

detect_objects_from_image_tool = tool(
    detect_objects_from_image,
    description="""
    카메라 이미지에서 객체들을 탐지하여 정보를 반환합니다.
    반환값 : [{class:객체명, box:[x1,y1,x2,y2]}, ...]
    반환값 중 box는 객체의 바운딩 박스를 나타내며, x1,y1은 왼쪽 위 꼭지점의 좌표, x2,y2는 오른쪽 아래 꼭지점의 좌표입니다. 
    """
)



tools = [
    # detect_objects_tool, 
    detect_objects_from_image_tool
    ]