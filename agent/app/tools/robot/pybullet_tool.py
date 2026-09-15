from langchain.tools import tool
from pydantic import BaseModel
from ...core.config import config
import requests
from typing import Any, Dict, List
import numpy as np
from ..vision.convert_pos import _pixel_to_camera, _camera_to_world


# ============ GET ============
def get_robot_state() -> dict[str, Any]:
    resp = requests.get(f"{config.BOT_URL}/robot_state") # {"ee": ee, "joints": joints}
    resp.raise_for_status()
    return resp.json()


class GetRobotStateCmd(BaseModel):
    pass

get_robot_state_tool = tool(
    get_robot_state,
    description=(
        "로봇팔의 현재 상태를 반환합니다. "
        "반환값: ee(손끝 위치, 단위 m, {'x':float,'y':float,'z':float}), "
        "joints(5개 관절 각도(deg)와 마지막 6번째는 그리퍼 상태값(0.0~0.06)을 포함한 총 6개의 요소 리스트, 예: [0.0, 1.57, -1.57, 0.0, 1.57, 0.06])"
    ),
    args_schema=GetRobotStateCmd
)



def get_object_state() -> dict[str, Any]:
    resp = requests.get(f"{config.BOT_URL}/object_state")
    resp.raise_for_status()
    return resp.json()


class GetObjectStateCmd(BaseModel):
    pass

get_object_state_tool = tool(
    get_object_state,
    description=(
        "씬에 있는 오브젝트의 현재 정보를 반환합니다. "
        "반환값: exists(존재 여부 bool), x/y/z(월드 좌표 float), "
        "distance(카메라에서 오브젝트까지의 직선 거리 float)"
    ),
    args_schema=GetObjectStateCmd
)






class GetImageCmd(BaseModel):
    pass

def _get_image() -> bytes:
    resp = requests.get(f"{config.BOT_URL}/image")
    resp.raise_for_status()
    return resp.content


def _get_depth() -> List[Any]:
    resp = requests.get(f"{config.BOT_URL}/depth")
    resp.raise_for_status()
    return resp.json()



# ============ yolo로 추론한 오브젝트의 중앙 좌표로 depth 거리 반환 ============
def _get_depth_value(depth, x, y, k=2):
    depth = np.array(depth)
    h, w = depth.shape

    x = int(np.clip(x, 0, w-1))
    y = int(np.clip(y, 0, h-1))

    patch = depth[y-k:y+k+1, x-k:x+k+1]

    valid = patch[patch > 0]

    if len(valid) == 0:
        return None

    return float(np.mean(valid))



# ============ 카메라로부터 이미지의 중앙 좌표 반환  ============
def _get_camera():
    """
    오브젝트를 잡기 위한 초기 단계: 이미지 가져오기, distance 가져오기, 객체 탐지 및 중앙 좌표 계산.
    """
    # 1. pybullet_tool.py의 _get_image() 함수로 이미지를 가져옴
    img = _get_image()

    # 2. _get_depth() 함수로 depth 정보를 가져옴
    depth = _get_depth()

    # 3. config.VISION_URL + "/detect_from_image" 경로에 Post 요청을 보냄 (body는 이미지)
    files = {
        'data': ('image.jpg', img, 'image/jpeg')
    }
    response = requests.post(
        config.VISION_URL + "/detect_from_image",
        files=files
    )
    detections = response.json()

    center_x = None
    center_y = None

    # 4. 반환받은 오브젝트의 중앙 좌표를 대입
    if detections["objects"]:
        # 첫 번째 오브젝트의 바운딩 박스 사용 (가정)
        box = detections["objects"][0]['xywh']  # [x, y, w, h]
        center_x = int(box[0])  # 픽셀단위
        center_y = int(box[1])  # 픽셀단위

        # yolo로 추론한 오브젝트의 중앙 좌표의 depth 데이터 추출
        center_z = _get_depth_value(depth, center_x, center_y)
        h, w = np.array(depth).shape

        return {'center_x': center_x, 'center_y': center_y, "center_z": center_z, "cam_w": w, "cam_h": h}
    else:
        return None



# ============ 월드 좌표계 기준 오브젝트 위치 반환 ============
class GetVisionObjectPosCmd(BaseModel):
    pass

def get_vision_object_pos() -> dict:
    object_xyz = _get_camera()
    if object_xyz is None: return {"error": "오브젝트를 찾지 못했습니다."}

    cam_w = object_xyz.pop("cam_w", 480)
    cam_h = object_xyz.pop("cam_h", 480)
    pc = _pixel_to_camera(**object_xyz, cam_width=cam_w, cam_height=cam_h)
    if pc is None: return {"error": "카메라 좌표 변환 실패."}

    pw = _camera_to_world(pc, camera_pos=[0.5, 0, 0.5], target=[0,0,0], up=[0,0,1])

    return {"x": float(pw[0]), "y": float(pw[1]), "z": float(pw[2])}

get_vision_object_pos_tool = tool(
    get_vision_object_pos,
    description="""
    카메라 이미지(Vision)를 바탕으로 씬에 있는 주요 객체를 탐지하고 해당 객체의 월드 좌표계 기준 절대 좌표(x, y, z)를 계산하여 반환합니다.
    (주의: 시각 추론 기반이므로 실제 위치와 약간의 오차가 존재할 수 있습니다.)
    반환값: {"x": float, "y": float, "z": float}
    """,
    args_schema=GetVisionObjectPosCmd
)







# ============ POST ============
class SetPosCmd(BaseModel):
    pos: List[float]  # [x, y, z]

def set_pos(pos: List[float]) -> Dict[str, Any]:
    resp = requests.post(f"{config.BOT_URL}/set_pos", json={"pos": pos})
    resp.raise_for_status()
    return resp.json()

set_pos_tool = tool(
    set_pos,
    description="""
    로봇의 말단을 지정된 위치로 이동시킵니다.
    입력값 pos는 [x, y, z] 형식의 리스트로, 단위는 미터(m)입니다.
    """,
    args_schema=SetPosCmd
)



class SetJointsCmd(BaseModel):
    joints: List[float]

def set_joints(joints: List[float]) -> Dict[str, Any]:
    resp = requests.post(f"{config.BOT_URL}/set_joints", json={"joints": joints})
    resp.raise_for_status()
    return resp.json()

set_joints_tool = tool(
    set_joints,
    description="""
    로봇의 관절을 지정된 각도로 이동시킵니다.
    입력값 joints는 5개 관절의 각도(deg)를 나타낸 리스트여야 합니다.
    예시: [0.0, 1.57, -1.57, 0.0, 1.57]
    """,
    args_schema=SetJointsCmd
)


class SetGripperCmd(BaseModel):
    value: float # 0.0 ~ 0.06

def set_gripper(value: float) -> Dict[str, Any]:
    resp = requests.post(f"{config.BOT_URL}/set_gripper", json={"gripper": value})
    resp.raise_for_status()
    return resp.json()

set_gripper_tool = tool(
    set_gripper,
    description="""
    그리퍼를 열거나 닫습니다.
    입력값 value는 그리퍼의 상태를 나타냅니다.
    0.0은 완전히 닫힌 상태, 0.06은 완전히 열린 상태를 나타냅니다.
    """,
    args_schema=SetGripperCmd
)


tools = [
    get_robot_state_tool,
    get_object_state_tool, # vision 모듈 없이 테스트할 경우 사용
    set_pos_tool,
    set_joints_tool,
    set_gripper_tool,
    # get_vision_object_pos_tool
]