import numpy as np
import cv2
import cv2.aruco as aruco
from .camera import get_intrinsics, get_aligned_frames
from .yolo_detect import detect_objects

CAMERA_POS = [0.70, 0.0, 0.55]  # Fallback: 로봇 월드 좌표계 기준 카메라의 렌즈 위치 (X, Y, Z)
BASE_POS = 0.0 # 마커 기준 로봇 베이스 위치

# 추적 안정화(Ambiguity Flip 방지)를 위한 이전 프레임 상태 저장
_prev_rvec = None

# ============ ArUco 마커 초기화 ============
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
detector = aruco.ArucoDetector(aruco_dict)

marker_length = 0.03  # 3cm
half = marker_length / 2
obj_points = np.array([
    [-half,  half, 0],
    [ half,  half, 0],
    [ half, -half, 0],
    [-half, -half, 0]
], dtype=np.float32)

def draw_custom_axes(img, K, dist, rvec, tvec, length):
    # 빨간색(X)을 원래 방향(오른쪽)으로 되돌리고, 초록색(Y)은 반대 방향을 유지합니다.
    pts = np.array([
        [0, 0, 0],
        [length, 0, 0],   # 빨강 (반대 방향 뒤집기 -> 오른쪽)
        [0, -length, 0],  # 초록 (반대 방향)
        [0, 0, length]    # 파랑 (유지)
    ], dtype=np.float32)

    imgpts, _ = cv2.projectPoints(pts, rvec, tvec, K, dist)
    imgpts = np.int32(imgpts).reshape(-1, 2)

    origin = tuple(imgpts[0])
    pt_x = tuple(imgpts[1])
    pt_y = tuple(imgpts[2])
    pt_z = tuple(imgpts[3])

    cv2.line(img, origin, pt_x, (0, 0, 255), 2) # 빨강
    cv2.line(img, origin, pt_y, (0, 255, 0), 2) # 초록
    cv2.line(img, origin, pt_z, (255, 0, 0), 2) # 파랑


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
    corners, ids, _ = detector.detectMarkers(gray)
    
    if ids is not None and len(ids) > 0:
        # 우선 첫 번째 인식된 마커를 원점 기준으로 삼습니다
        img_points = corners[0][0]
        
        # [Z축 뒤집힘 방지] 평면 마커의 모호성(Ambiguity)을 해결하기 위해 다중 해를 구합니다.
        try:
            success, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj_points, img_points, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if success and len(rvecs) > 0:
                rvec, tvec = rvecs[0], tvecs[0]
                
                if len(rvecs) > 1:
                    if _prev_rvec is not None:
                        # 통계적으로 가장 확실한 제어: 이전 프레임의 자세와 동일/유사한 해답을 선택
                        dist0 = min(np.linalg.norm(rvecs[0] - _prev_rvec), np.linalg.norm(rvecs[0] + _prev_rvec))
                        dist1 = min(np.linalg.norm(rvecs[1] - _prev_rvec), np.linalg.norm(rvecs[1] + _prev_rvec))
                        
                        if dist1 < dist0:
                            rvec, tvec = rvecs[1], tvecs[1]
                    else:
                        # 초기 인식 시: 카메라와 Z축이 마주보는 방향을 우선 선택
                        R_0, _ = cv2.Rodrigues(rvecs[0])
                        R_1, _ = cv2.Rodrigues(rvecs[1])
                        if np.dot(tvecs[0].flatten(), R_0[:, 2]) > 0 and np.dot(tvecs[1].flatten(), R_1[:, 2]) < 0:
                            rvec, tvec = rvecs[1], tvecs[1]
                
                # 부호 동일화(Rodrigues 회전 벡터 특성 보정) 후 저장
                if _prev_rvec is not None and np.linalg.norm(rvec + _prev_rvec) < np.linalg.norm(rvec - _prev_rvec):
                    rvec = -rvec
                _prev_rvec = rvec.copy()
                        
                R_cm, _ = cv2.Rodrigues(rvec)
            else:
                success = False
        except AttributeError:
            success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist)
            if success:
                if _prev_rvec is not None and np.linalg.norm(rvec + _prev_rvec) < np.linalg.norm(rvec - _prev_rvec):
                    rvec = -rvec
                _prev_rvec = rvec.copy()
                R_cm, _ = cv2.Rodrigues(rvec)

        if success:
            
            # [원점 보정] X좌표 값으로 2cm 뒤를 새로운 원점(0,0,0)으로 설정
            # World의 X(앞/뒤) = Marker의 -Y 에 해당합니다.
            # World X축으로 -2cm(-0.02m) 이동한 위치가 새 원점이 되려면,
            # Marker 기준으로는 Y축 방향으로 +0.02m 이동해야 합니다.
            offset_marker = np.array([[0.0], [BASE_POS], [0.0]], dtype=np.float32)
            t_cm_new = R_cm @ offset_marker + tvec
            
            t_cm = t_cm_new
            use_marker = True
            
            if return_image:
                draw_custom_axes(rgb_image, K, dist, rvec, t_cm, 0.05)

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
                
                p_world_x = -my  # X: 앞 (Marker's -Y)
                p_world_y = mx   # Y: 반전 적용 (Marker's X)
                p_world_z = mz   # Z: 상 (Marker's Z)
                
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
                "world_x": p_world_x, 
                "world_y": p_world_y, 
                "world_z": p_world_z  
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
                text2 = f"X:{p_world_x:.2f} Y:{p_world_y:.2f} Z:{p_world_z:.2f}"
                cv2.putText(rgb_image, text1, (x1, max(y1 - 22, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(rgb_image, text2, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if return_image:
        return world_objects, rgb_image
        
    return world_objects


def get_objects_world_pos():
    return get_world_coordinates()