import numpy as np
import cv2
import cv2.aruco as aruco
from .camera import get_intrinsics, get_aligned_frames
from .yolo_detect import detect_objects

CAMERA_POS = [0.70, 0.0, 0.55]  # Fallback: 로봇 월드 좌표계 기준 카메라의 렌즈 위치 (X, Y, Z)
BASE_POS = -0.10 # 마커 기준 로봇 베이스 위치
MARKER_SIZE = 0.04

# 추적 안정화(Ambiguity Flip 방지)를 위한 이전 프레임 상태 저장
_prev_rvec = None

# ============ ArUco 마커 초기화 ============
# OpenCV 버전 호환성 대응
if hasattr(cv2.aruco, 'getPredefinedDictionary'):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    use_detector = True
else:
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters_create()
    use_detector = False

half = MARKER_SIZE / 2

# test_realsense_aruco.py 와 동일한 마커 3D 좌표 기준 사용
obj_points = np.array([
    [-half,  half, 0],
    [ half,  half, 0],
    [ half, -half, 0],
    [-half, -half, 0]
], dtype=np.float32)

def draw_custom_axes(img, K, dist, rvec, tvec, length):
    # X축(빨강)만 반대 방향(-X)으로 표시, Y(초록)와 Z(파랑)는 그대로 유지
    pts = np.array([
        [0, 0, 0],
        [-length, 0, 0],   # 빨강 (-X 방향)
        [0, length, 0],    # 초록 (+Y 방향)
        [0, 0, length]     # 파랑 (+Z 방향)
    ], dtype=np.float32)

    imgpts, _ = cv2.projectPoints(pts, rvec, tvec, K, dist)
    imgpts = np.int32(imgpts).reshape(-1, 2)

    origin = tuple(imgpts[0])
    pt_x = tuple(imgpts[1])
    pt_y = tuple(imgpts[2])
    pt_z = tuple(imgpts[3])

    thickness = 2 # 선의 굵기를 반으로 줄임
    cv2.line(img, origin, pt_x, (0, 0, 255), thickness) # 빨강
    cv2.line(img, origin, pt_y, (0, 255, 0), thickness) # 초록
    cv2.line(img, origin, pt_z, (255, 0, 0), thickness) # 파랑



# ============ 1. 카메라 픽셀, 깊이 및 RGB 가져오기 ============
def _get_object_pos(target_class=None):
    aligned_frames = get_aligned_frames()
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    
    if not color_frame or not depth_frame:
        print("Error: 카메라에서 프레임을 가져오지 못했습니다.")
        return [], None

    rgb_image = np.asanyarray(color_frame.get_data())
    depth_data = np.asanyarray(depth_frame.get_data())
    
    detected_objects = detect_objects(rgb_image)

    valid_objects = []
    height, width = depth_data.shape
    
    if target_class:
        detected_objects = [obj for obj in detected_objects if obj["class_name"] == target_class]
        
    for obj in detected_objects:
        cx, cy, w, h = obj["xywh"]
        cxi, cyi = int(cx), int(cy)
        
        if 0 <= cyi < height and 0 <= cxi < width:
            depth_value = float(depth_data[cyi, cxi])
        else:
            depth_value = 0.0

        if depth_value <= 0:
            continue

        valid_objects.append({
            "class_name": obj["class_name"],
            "center_x": cx,
            "center_y": cy,
            "w": w,
            "h": h,
            "center_z": depth_value / 1000.0  # 밀리미터 -> 미터 변환
        })

    return valid_objects, rgb_image



def get_world_coordinates(target_class=None, camera_pos=tuple(CAMERA_POS), return_image=False):
    """
    탐지된 객체의 픽셀 좌표와 Depth를 마커 기준의 월드 좌표계로 변환합니다.
    마커가 감지되지 않으면 IMU와 camera_pos를 사용하는 Fallback 모드로 동작합니다.
    return_image=True 설정 시, 시각화 마커와 좌표값이 합성된 이미지가 함께 반환됩니다.
    """
    global _prev_rvec
    
    objects, rgb_image = _get_object_pos(target_class)
    if rgb_image is None:
        if return_image: return [], None
        return []

    intrinsics = get_intrinsics()
    if not intrinsics:
        print("Error: 카메라 파라미터를 가져오지 못했습니다.")
        if return_image: return [], rgb_image
        return []

    fx, fy = intrinsics["fx"], intrinsics["fy"]
    ppx, ppy = intrinsics["ppx"], intrinsics["ppy"]

    # intrinsic matrix K
    K = np.array([
        [fx, 0, ppx],
        [0, fy, ppy],
        [0, 0, 1]
    ], dtype=np.float32)
    dist = np.zeros((5,1))

    # ============ 2. 마커 기준 좌표 변환 행렬 도출 ============
    use_marker = False
    R_cm = None 
    t_cm = None
    
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    
    if use_detector:
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    if ids is not None and len(ids) > 0:
        # 우선 첫 번째 인식된 마커를 기반으로 처리
        img_points = corners[0][0]
        
        # test_realsense_aruco.py 와 같이 단순 solvePnP 사용
        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist)
        
        if success:
            R_cm, _ = cv2.Rodrigues(rvec)
            
            # 무조건 Z축이 수직 위(카메라를 향하는 방향)로 오도록 강제 플립
            # 카메라에서 마커로의 시선 벡터(tvec)와 마커의 Z축(R_cm[:, 2]) 내적이 양수면 Z축이 바닥을 뚫는 상태임
            if np.dot(tvec.flatten(), R_cm[:, 2]) > 0:
                # X축을 기준으로 180도 회전하여 Y축과 Z축의 방향을 뒤집음
                R_cm = R_cm @ np.array([[1, 0, 0], 
                                        [0, -1, 0], 
                                        [0, 0, -1]], dtype=np.float32)
                rvec, _ = cv2.Rodrigues(R_cm)
            
            # [원점 보정] BASE_POS(10cm 등)를 새로운 원점으로 설정하기 위한 오프셋 변환
            # 모델링한 좌표계에서 베이스로 이동하기 위해 (Offset 추가)
            # 여기서는 월드의 x,y,z 스펙에 따라 오프셋 위치가 달라질 수 있으므로 기존 위치를 유지하나
            # y축 보정 등은 마커의 축에 따름 (기존 로직: 10cm 오프셋)
            offset_marker = np.array([[0.0], [BASE_POS], [0.0]], dtype=np.float32)
            t_cm_new = R_cm @ offset_marker + tvec
            
            t_cm = t_cm_new
            use_marker = True
            
            if return_image:
                aruco.drawDetectedMarkers(rgb_image, corners, ids)
                # X축 반전, 굵기 반감, 길이 연장을 위한 커스텀 함수 호출
                draw_custom_axes(rgb_image, K, dist, rvec, t_cm, MARKER_SIZE * 1.5)

    # Fallback 로직용 변수
    R_imu = None
    if not use_marker:
        pitch_deg = intrinsics["pitch"]
        roll_deg = intrinsics["roll"]

        adjusted_roll_deg = roll_deg + 90.0
        theta_x = np.radians(pitch_deg)       
        theta_y = np.radians(adjusted_roll_deg) 

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
        R_imu = Ry @ Rx
        cam_tx, cam_ty, cam_tz = camera_pos
        
    world_objects = []
    
    # ============ 3. 오브젝트 좌표 변환 ============
    if objects:
        for obj in objects:
            cx = obj["center_x"]
            cy = obj["center_y"]
            z_cam = obj["center_z"] 
            
            # 픽셀 -> 카메라 기준 3D 투영
            x_cam = (cx - ppx) * z_cam / fx
            y_cam = (cy - ppy) * z_cam / fy
            
            if use_marker:
                P_cam = np.array([[x_cam], [y_cam], [z_cam]], dtype=np.float32)
                
                # P_world = R^-1 * (P_cam - t)
                P_marker = R_cm.T @ (P_cam - t_cm)
                
                mx, my, mz = P_marker.flatten()
                
                # test 스크립트 좌표계 방식(cv2 표준)으로 객체 좌표 도출
                # X: Red(Right/Left), Y: Green(Up/Down), Z: Blue(Outward)
                # 따라서 로봇 좌표계 변환 방식도 이 축에 기반하여 재설정합니다.
                
                p_world_x = my   # 앞/뒤 (사용자의 로봇 축 사양에 맞게 조정)
                p_world_y = -mx  # 좌/우
                p_world_z = mz   # 상/하
                
            else: # Fallback
                x_base = z_cam    
                y_base = -x_cam   
                z_base = -y_cam   
                
                p_base = np.array([x_base, y_base, z_base])
                p_rotated = R_imu @ p_base
                p_yawed = np.array([-p_rotated[0], -p_rotated[1], p_rotated[2]])
                
                p_world = p_yawed + np.array([cam_tx, cam_ty, cam_tz])
                
                p_world_x = p_world[0]
                p_world_y = p_world[1]  
                p_world_z = p_world[2]

            world_objects.append({
                "class_name": obj["class_name"],
                "world_x": round(p_world_x, 4), 
                "world_y": round(p_world_y, 4), 
                "world_z": round(p_world_z, 4)  
            })
            
            if return_image:
                w, h = obj["w"], obj["h"]
                x1 = int(cx - w / 2)
                y1 = int(cy - h / 2)
                x2 = int(cx + w / 2)
                y2 = int(cy + h / 2)
                
                # Bounding box
                cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Info Text
                text1 = f"{obj['class_name']}"
                text2 = f"X:{p_world_x:.3f} Y:{p_world_y:.3f} Z:{p_world_z:.3f}"
                cv2.putText(rgb_image, text1, (x1, max(y1 - 22, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(rgb_image, text2, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if return_image:
        return world_objects, rgb_image
        
    return world_objects


def get_objects_world_pos():
    return get_world_coordinates()