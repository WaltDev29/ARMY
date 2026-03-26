import numpy as np
from .camera import get_detection_data, get_intrinsics

CAMERA_POS = [0.70, 0.0, 0.55]  # 로봇 월드 좌표계 기준 카메라의 렌즈 위치 (X, Y, Z)


# ============ 1. 카메라 픽셀 및 깊이 가져오기 ============
def _get_object_pos(target_class=None):
    detections = get_detection_data()
    
    if isinstance(detections, dict) and "error" in detections:
        print(f"Vision API Error: {detections.get('error')}")
        return []

    if not detections:
        return []

    if target_class:
        detections = [obj for obj in detections if obj["class_name"] == target_class]
        
    if not detections:
        return []

    valid_objects = []
    for obj in detections:
        cx, cy, _, _ = obj["xywh"]
        dist_mm = obj["distance_mm"]

        if dist_mm <= 0:
            continue

        valid_objects.append({
            "class_name": obj["class_name"],
            "center_x": cx,
            "center_y": cy,
            "center_z": dist_mm / 1000.0  # 밀리미터 -> 미터 변환
        })

    return valid_objects



def get_world_coordinates(target_class=None, camera_pos=(0.0, 0.0, 0.0)):
    """
    탐지된 객체의 픽셀 좌표와 Depth를 월드 좌표계로 변환합니다.
    
    :param target_class: 탐지할 객체의 클래스 이름
    :param camera_pos: 월드 원점 기준 카메라의 현재 위치 (X, Y, Z) 미터 단위 튜플
    """
    # 카메라 위치 (Translation) 추출
    cam_tx, cam_ty, cam_tz = camera_pos

    objects = _get_object_pos(target_class)
    if not objects:
        print("Error: 탐지된 객체가 없습니다.")
        return []
        
    intrinsics = get_intrinsics()
    if not intrinsics:
        print("Error: 카메라 파라미터를 가져오지 못했습니다.")
        return []

    fx, fy = intrinsics["fx"], intrinsics["fy"]
    ppx, ppy = intrinsics["ppx"], intrinsics["ppy"]
    pitch_deg = intrinsics["pitch"]
    roll_deg = intrinsics["roll"]

    # 각도 보정 및 라디안 변환 (이전과 동일)
    adjusted_roll_deg = roll_deg + 90.0
    
    theta_x = np.radians(pitch_deg)       
    theta_y = np.radians(adjusted_roll_deg) 

    # 회전 행렬 (Rotation Matrix) 생성
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])
    
    Ry = np.array([
        [np.cos(theta_y), 0, -np.sin(theta_y)],
        [0, 1, 0],
        [np.sin(theta_y), 0, np.cos(theta_y)]
    ])
    
    R = Ry @ Rx

    world_objects = []
    for obj in objects:
        cx = obj["center_x"]
        cy = obj["center_y"]
        z_cam = obj["center_z"] 
        
        # 1. 픽셀 -> 카메라 좌표계
        x_cam = (cx - ppx) * z_cam / fx
        y_cam = (cy - ppy) * z_cam / fy
        
        # 2. 카메라 좌표계 -> 기본 월드 축 (일단 +X 정면을 본다고 가정)
        x_base = z_cam    
        y_base = -x_cam   
        z_base = -y_cam   
        
        p_base = np.array([x_base, y_base, z_base])
        
        # 3. 회전 행렬 적용 (IMU 기반 기울기 보정)
        p_rotated = R @ p_base
        
        # 4. [핵심 수정] 카메라가 원점(-X 방향)을 향해 180도 뒤돌아 있는 상태 반영
        # Z축(높이/상하)은 그대로 두고 X(앞뒤), Y(좌우)의 방향만 반전시킵니다.
        p_yawed = np.array([-p_rotated[0], -p_rotated[1], p_rotated[2]])
        
        # 5. 카메라 위치(Translation) 합산 (최종 절대 좌표)
        p_world = p_yawed + np.array([cam_tx, cam_ty, cam_tz])
        
        world_objects.append({
            "class_name": obj["class_name"],
            "world_x": p_world[0], 
            "world_y": p_world[1], 
            "world_z": p_world[2]  
        })
        
    return world_objects



def get_objects_world_pos():
    return get_world_coordinates(camera_pos=CAMERA_POS)