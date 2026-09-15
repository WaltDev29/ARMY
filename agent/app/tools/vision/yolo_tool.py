from langchain.tools import tool
from ...core.config import config
import requests
from pydantic import BaseModel
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR.parent.parent / "memory/images"



# ============ Vision API에 이미지를 보내 객체 탐지하는 Tool ============ 

def detect_objects_from_image():
    # 1. 시뮬레이터(또는 로봇)에서 현재 카메라 이미지를 캡처
    try:
        img_resp = requests.get(f"{config.BOT_URL}/image", timeout=5.0)
        img_resp.raise_for_status()
        img_bytes = img_resp.content
    except Exception as e:
        return {"error": f"이미지 캡처 실패: {str(e)}"}

    # 2. 캡처한 이미지를 Vision 서버로 전송하여 객체 탐지
    files = {
        'data': ('image.jpg', img_bytes, 'image/jpeg')
    }

    try:
        response = requests.post(
            f"{config.VISION_URL}/detect_from_image",
            files=files,
            timeout=10.0
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": f"객체 탐지 요청 실패: {str(e)}"}

detect_objects_from_image_tool = tool(
    detect_objects_from_image,
    description="""
    현재 카메라 화면을 캡처하여 화면 내의 객체들을 탐지하고 정보를 반환합니다. 인자 없이 호출하세요.
    반환값 : {"objects": [{class:객체명, box:[x1,y1,x2,y2], xywh:[x,y,w,h]}, ...]}
    반환값 중 box는 객체의 바운딩 박스를 나타내며, x1,y1은 왼쪽 위 꼭지점의 좌표, x2,y2는 오른쪽 아래 꼭지점의 좌표입니다.
    반환값 중 xywh는 객체의 중심 좌표 x,y와 물체의 너비 w, 높이 h입니다.
    """
)



tools = [
    # detect_objects_from_image_tool # 비전 Tool 사용 테스트용
    ]