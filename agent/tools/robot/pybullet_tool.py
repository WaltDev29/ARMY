from langchain.tools import tool
from pydantic import BaseModel
from ...core.config import config
import requests
from typing import Any, Dict, List
import numpy as np
from ..vision.convert_pos import _pixel_to_camera, _camera_to_world


# ============ GET ============
def get_image() -> bytes:
    resp = requests.get(f"{config.BOT_URL}/image")
    resp.raise_for_status()
    return resp.content


get_image_tool = tool(
    get_image,
    description="""
    카메라에서 이미지를 반환받습니다.
    """
)


def get_depth() -> List[Any]:
    resp = requests.get(f"{config.BOT_URL}/depth")
    resp.raise_for_status()
    return resp.json()


def get_robot_state() -> dict[str, Any]:
    resp = requests.get(f"{config.BOT_URL}/robot_state") # {"ee": ee, "joints": joints}
    resp.raise_for_status()
    return resp.json()


get_robot_state_tool = tool(
    get_robot_state,
    description="""
    # 로봇팔의 현재 상태를 반환합니다.

    ## 반환값 형식
    {"ee": ee, "joints": joints}

    ## 반환값 설명
    1. ee : 로봇팔의 손(그리퍼) 끝 위치
        - 반환값은 {'x': float, 'y': float, 'z': float} 형식의 딕셔너리입니다.
        - 단위는 미터(m)입니다.
    2. 로봇팔의 현재 관절 각도
        - 각 관절의 각도(deg)를 나타낸 리스트입니다.
        - 예시: [0.0, 1.57, -1.57, 0.0, 1.57, 0.0]
    """
)



def get_object_state() -> dict[str, Any]:
    resp = requests.get(f"{config.BOT_URL}/object_state")
    resp.raise_for_status()
    return resp.json()


get_object_state_tool = tool(
    get_object_state,
    description="""
    # 오브젝트의 정보를 반환합니다.

    ## 반환값 형식
    object : {
        "exists": bool,
        "x": float, "y": float, "z": float,
        "distance": float
    }

    ## 반환값 설명
    exists : 오브젝트의 존재 여부
    x, y, z : 오브젝트의 위치 좌표
    distance : 카메라에서 오브젝트까지의 직선 거리
    """
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
    입력값 joints는 각 관절의 각도(deg)를 나타낸 리스트입니다.
    예시: [0.0, 1.57, -1.57, 0.0, 1.57, 0.0]
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



# ============ Robot 힘, 속도 제어 (DOFBot에 미구현) ============
def set_force(force: float) -> Dict[str, Any]:
    resp = requests.post(f"{config.BOT_URL}/force", json={"force": force})
    resp.raise_for_status()
    return resp.json()


def set_max_velocity(max_vel: float) -> Dict[str, Any]:
    resp = requests.post(f"{config.BOT_URL}/max_velocity", json={"max_velocity": max_vel})
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
    # 1. pybullet_tool.py의 get_image() 함수로 이미지를 가져옴
    img = get_image()

    # 2. get_depth() 함수로 depth 정보를 가져옴
    depth = get_depth()

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

        return {'center_x': center_x, 'center_y': center_y, "center_z": center_z}
    else:
        return None







# ============ 월드 좌표계 기준 오브젝트 위치 반환 ============
def _get_object_pos():
    object_xyz = _get_camera()
    if object_xyz is None: return None

    pc = _pixel_to_camera(**object_xyz, cam_width=720, cam_height=720)
    if pc is None: return None

    pw = _camera_to_world(pc, camera_pos=[0.5, 0, 0.5], target=[0,0,0], up=[0,0,1])

    return pw



def grab_object():
    """
    오브젝트를 탐지하여 로봇팔을 해당 좌표로 이동한 후 그리퍼를 닫습니다.
    """
    set_gripper(0.06)

    # ============ Loop ============
    while True:
        # ============ GET 오브젝트 정보 ============
        object_pos = _get_object_pos()
        if object_pos is None: break
        object_x, object_y, object_z = object_pos
        print(f"object_pos : [{object_x}, {object_y}, {object_z}]")

        # 오브젝트 위치 && 로봇 위치 비교
        robot_state = get_robot_state()
        if not abs(robot_state.get("ee")["x"] - object_x) >= 0.03 and not abs(robot_state.get("ee")["y"] - object_y) >= 0.03 and not abs(robot_state.get("ee")["z"] - object_z) >= 0.03: break
        print(f"robot_pos : [{robot_state.get('ee')['x']}, {robot_state.get('ee')['y']}, {robot_state.get('ee')['z']}]")

        # 로봇 이동 명령
        robot_resp = set_pos([object_x, object_y, object_z])
        if not robot_resp.get("ok"): return False

    # ============ 그리퍼 닫기 ============
    set_gripper(0.0)

    return True


grab_object_tool = tool(
    grab_object,
    description="""
    오브젝트의 위치를 파악하여 로봇팔로 오브젝트를 잡습니다.
    """
)


tools = [
    get_robot_state_tool,
    get_object_state_tool,
    set_pos_tool,
    # set_joints_tool,
    set_gripper_tool,
    grab_object_tool
]