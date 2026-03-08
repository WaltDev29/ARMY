from langchain.tools import tool
from pydantic import BaseModel
from ...core.config import config
import requests
from typing import Any, Dict, List



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


tools = [
    get_robot_state_tool,
    get_object_state_tool,
    set_pos_tool,
    # set_joints_tool,
    set_gripper_tool
]