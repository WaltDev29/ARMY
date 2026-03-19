from langchain.tools import tool
from pydantic import BaseModel
from ...core.config import config
import requests
from typing import Any, Dict, List
from ..vision.convert_pos import _pixel_to_camera, _camera_to_world
from ..vision.realsense_tool import get_realsense_detections


# ============ GET ============
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



# ============ POST ============
# Set Pos
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



# Set Gripper
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




# ============ 월드 좌표계 기준 오브젝트 위치 반환 ============
def _get_object_pos(target_class=None):
    # 1. RealSense API를 통해 객체 탐지 결과를 가져옴
    response = get_realsense_detections()
    if response.get("status") == "error":
        print(f"Vision API Error: {response.get('error')}")
        return None

    detections = response.get("detections", [])
    
    if not detections:
        return None

    # target_class 지정 시 해당 객체만 탐색, 없으면 첫 번째 객체
    target_obj = None
    if target_class:
        for obj in detections:
            if obj["class_name"] == target_class:
                target_obj = obj
                break
    else:
        target_obj = detections[0]
        
    if not target_obj:
        return None

    cx, cy, _, _ = target_obj["xywh"]
    dist_mm = target_obj["distance_mm"]

    # 거리 값이 유효하지 않은 경우 무시
    if dist_mm <= 0:
        return None

    # 2. 중심 좌표 및 거리 전달 (거리는 밀리미터 -> 미터 단위 변환)
    object_xyz = {
        'center_x': cx,
        'center_y': cy,
        'center_z': dist_mm / 1000.0
    }

    # 카메라 내부 파라미터(Intrinsics) 추출
    intrinsics = response.get("intrinsics")
    if intrinsics:
        cam_width = intrinsics.get("width", 640)
        cam_height = intrinsics.get("height", 480)
        fx = intrinsics.get("fx")
        fy = intrinsics.get("fy")
        int_cx = intrinsics.get("ppx")
        int_cy = intrinsics.get("ppy")
        pc = _pixel_to_camera(**object_xyz, cam_width=cam_width, cam_height=cam_height, fx=fx, fy=fy, cx=int_cx, cy=int_cy)
    else:
        # 3. 픽셀 좌표를 카메라 좌표계로 변환 (카메라 해상도 640x480 고려, 근사치 연산)
        pc = _pixel_to_camera(**object_xyz, cam_width=640, cam_height=480)
        
    if pc is None: return None

    # 4. 카메라 좌표계를 월드 좌표계로 변환 (로봇 월드 중심 기준)
    pw = _camera_to_world(pc, camera_pos=[0.5, 0, 0.5], target=[0,0,0], up=[0,0,1])

    return pw



def grab_object(target_class: str = None):
    """
    오브젝트를 탐지하여 로봇팔을 해당 좌표로 이동한 후 그리퍼를 닫습니다.
    입력값 target_class를 전달하면 특정 객체만 목표로 잡을 수 있습니다.
    """
    set_gripper(0.06)

    # ============ Loop ============
    while True:
        # ============ GET 오브젝트 정보 ============
        object_pos = _get_object_pos(target_class)
        if object_pos is None: 
            print("물체를 찾지 못했습니다.")
            set_gripper(0.0) # 혹시라도 그리퍼가 열린 상태에서 실패하면 닫아둠
            return False
            
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
    카메라를 통해 오브젝트의 3D 월드 좌표를 계산하고, 로봇팔을 해당 위치로 이동시켜 그리퍼로 물체를 잡는(Pick) 작업을 수행합니다.
    입력값 `target_class`로 특정 물체의 이름(예: "bottle", "person", "cup")을 문자열로 전달하면 해당 물체만 골라서 잡을 수 있습니다.
    비워두면 화면에 보이는 첫 번째 물체를 잡습니다.
    물체를 파지하거나 조작하라는 명령이 있을 때 이 도구를 사용하세요.
    """
)


tools = [
    get_robot_state_tool,
    set_pos_tool,
    set_gripper_tool,
    grab_object_tool
]