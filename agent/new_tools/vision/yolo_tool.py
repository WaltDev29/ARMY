from langchain.tools import tool
from ...core.config import config
import requests
from pydantic import BaseModel
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR.parent.parent / "memory/images"



# ============ Vision API에 이미지를 보내 객체 탐지하는 Tool ============ 
class Image(BaseModel):
    img:bytes

def detect_objects_from_image(img:Image):
    files = {
        'data': ('image.jpg', img.img, 'image/jpeg')
    }

    response = requests.post(
        config.VISION_URL + "/detect_from_image",
        files=files
    )

    return response.json()

detect_objects_from_image_tool = tool(
    detect_objects_from_image,
    description="""
    주어진 이미지에서 객체들을 탐지하여 정보를 반환합니다.
    반환값 : [{class:객체명, box:[x1,y1,x2,y2], xywh:[x,y,w,h]}, ...]
    반환값 중 box는 객체의 바운딩 박스를 나타내며, x1,y1은 왼쪽 위 꼭지점의 좌표, x2,y2는 오른쪽 아래 꼭지점의 좌표입니다.
    반환값 중 xywh는 객체의 중심 좌표 x,y와 물체의 너비 w, 높이 h입니다.
    """,
    args_schema=Image
)



tools = [
    detect_objects_from_image_tool
    ]