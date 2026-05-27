from langchain.tools import tool
from pydantic import BaseModel
from ...core.config import config
import requests
from typing import Any, Dict, List


# ============ GET ============
def get_robot_state() -> dict[str, Any]:
    resp = requests.get(f"{config.BOT_URL}/robot/state") # {"ee": ee, "joints": joints}
    resp.raise_for_status()
    return resp.json()


get_robot_state_tool = tool(
    get_robot_state,
    description=(
        "로봇팔의 현재 상태를 반환합니다. "
        "반환값: ee(손끝 위치, 단위 m, {'x':float,'y':float,'z':float}), "
        "joints(각 관절 각도 deg 리스트, 예: [0.0, 1.57, -1.57, 0.0, 1.57, 0.0])"
        "joints의 마지막 값은 그리퍼의 각도입니다. 170은 닫힘, 10은 열림을 나타냅니다."
    )
)



# ============ POST ============
# Set Pos
class SetPosCmd(BaseModel):
    pos: List[float]  # [x, y, z]

def set_pos(pos: List[float]) -> Dict[str, Any]:
    resp = requests.post(f"{config.BOT_URL}/robot/set_pos", json={"pos": pos})
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



# Set Gripper
class SetGripperCmd(BaseModel):
    value: float # 10~170

def set_gripper(value: float) -> Dict[str, Any]:
    resp = requests.post(f"{config.BOT_URL}/robot/set_gripper", json={"gripper": value})
    resp.raise_for_status()
    return resp.json()

set_gripper_tool = tool(
    set_gripper,
    description="""
    그리퍼를 열거나 닫습니다.
    입력값 value는 그리퍼의 상태를 나타냅니다.
    170은 완전히 닫힌 상태, 10은 완전히 열린 상태를 나타냅니다.
    10 미만, 170 이상의 값은 오류를 반환합니다.
    """,
    args_schema=SetGripperCmd
)



# Set Torque
class SetTorqueCmd(BaseModel):
    value: int # 0,1

def set_torque(value: int) -> Dict[str, Any]:
    resp = requests.post(f"{config.BOT_URL}/robot/set_torque", json={"torque": value})
    resp.raise_for_status()
    return resp.json()

set_torque_tool = tool(
    set_torque,
    description="""
    로봇의 토크를 설정합니다.
    0은 해제, 1은 활성화입니다.
    """,
    args_schema=SetTorqueCmd
)




tools = [
    get_robot_state_tool,
    set_pos_tool,
    set_gripper_tool,
    set_torque_tool
]