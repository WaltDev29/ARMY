from langchain.tools import tool
from .pybullet_tool import get_robot_state, get_object_state, get_image, get_depth, set_pos, set_gripper
from ...core.config import config
import requests
import numpy as np


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



# ============ 카메라 좌표계 기준 오브젝트 좌표 반환  ============
def _pixel_to_camera(center_x, center_y, center_z):
    '''
    카메라 좌표계에서의 객체 위치 반환
    '''
    width, height = 720, 720
    fov = 60

    fx = width / (2 * np.tan(np.deg2rad(fov / 2)))
    fy = height / (2 * np.tan(np.deg2rad(fov / 2)))

    cx = width / 2
    cy = height / 2

    X = (center_x - cx) * center_z / fx
    Y = (center_y - cy) * center_z / fy
    Z = center_z

    return np.array([X, Y, Z])
    


# ============ 월드 좌표계 기준 오브젝트 좌표 반환  ============
def _camera_to_world(Pc, camera_pos, target, up):
    camera_pos = np.array(camera_pos)
    target = np.array(target)
    up = np.array(up)

    # 1. forward (카메라가 보는 방향)
    forward = target - camera_pos
    forward = forward / np.linalg.norm(forward)

    # 2. right
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    # 3. up 재계산 (직교화)
    true_up = np.cross(forward, right)

    # 4. 회전 행렬
    R = np.stack([right, true_up, forward], axis=1)

    # 5. 변환
    Pw = R @ Pc + camera_pos

    return Pw



# ============ 월드 좌표계 기준 오브젝트 위치 반환 ============
def _get_object_pos():
    object_xyz = _get_camera()
    if object_xyz is None: return None

    pc = _pixel_to_camera(**object_xyz)
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



tools = [grab_object_tool]