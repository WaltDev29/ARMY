from langchain.tools import tool
from ...core.config import config
import requests
from pydantic import BaseModel


class Detection(BaseModel):
    objects:list|None

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


tools = [detect_objects_tool]