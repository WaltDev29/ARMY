import numpy as np
import cv2
import cv2.aruco as aruco
from .camera import get_intrinsics, get_aligned_frames
from .detection_manager import detect_objects
from .segmentation_mask import generate_masks
import logging

logger = logging.getLogger(__name__)

CAMERA_POS = [0.70, 0.0, 0.55]  # Fallback: 로봇 월드 좌표계 기준 카메라의 렌즈 위치 (X, Y, Z)
BASE_POS = -0.0 # 마커 기준 로봇 베이스 위치
MARKER_SIZE = 0.03

# 떨림 방지 및 정밀도 향상을 위한 설정
Z_SCALING_FACTOR = 1.0     # 거리 비례 오차 보정
HISTORY_SIZE = 5            # 이동 평균 필터 크기 (클수록 부드럽지만 반응이 느려짐)
_object_history = {}        # { "class_name": [ (x, y, z), ... ] }

# ============ ArUco 마커 초기화 ============
# OpenCV 버전 호환성 대응
if hasattr(cv2.aruco, 'getPredefinedDictionary'):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    # 서브픽셀 검출 활성화 (거리가 멀 때의 정밀도 향상)
    aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    use_detector = True
else:
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters_create()
    aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
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
def _get_object_pos(target_classes=None):
    aligned_frames = get_aligned_frames()
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    
    if not color_frame or not depth_frame:
        print("Error: 카메라에서 프레임을 가져오지 못했습니다.")
        return [], None

    rgb_image = np.asanyarray(color_frame.get_data())
    depth_data = np.asanyarray(depth_frame.get_data())
    height, width = depth_data.shape
    
    # 1. 1차 시도: 표준 YOLOv11 (상시 모델)
    detected_objects = detect_objects(rgb_image)
    
    # target_classes가 지정된 경우 필터링 및 Fallback 체크
    if target_classes:
        # 대소문자 무시를 위해 클리닝
        cleaned_targets = [t.lower().strip() for t in target_classes]
        
        # 표준 모델에서 잡힌 타겟들
        standard_filtered = [obj for obj in detected_objects if obj["class_name"].lower().strip() in cleaned_targets]
        
        # 여전히 못 찾은 타겟(Missing Targets) 골라내기
        found_classes = {obj["class_name"].lower().strip() for obj in standard_filtered}
        missing_targets = [t for t in cleaned_targets if t not in found_classes]
        
        # 하나라도 못 찾은 게 있다면 YOLO-World (Fallback) 가동
        if missing_targets:
            # logger.info(f"YOLO-World 가동 (누락 타겟: {missing_targets})")
            world_objects = detect_objects(rgb_image, prompt=missing_targets)
            standard_filtered.extend(world_objects)
            
        detected_objects = standard_filtered

    if not detected_objects:
        return [], rgb_image

    # 2. 탐지된 객체들에 대해 FastSAM 마스크 생성
    bboxes = [obj["box_corner"] for obj in detected_objects]
    masks = generate_masks(rgb_image, bboxes)

    valid_objects = []
    for i, obj in enumerate(detected_objects):
        mask = masks[i]
        cx, cy, w, h = obj["xywh"]
        
        # 1. 마스크가 있는 경우: 무게 중심(Centroid) 및 정제된 Depth 계산
        if mask is not None:
            # 픽셀 좌표 리스트 추출
            y_indices, x_indices = np.where(mask)
            if len(x_indices) > 0:
                # 바운딩 박스 중심 대신 실제 마스크의 무게 중심 사용
                cx = float(np.mean(x_indices))
                cy = float(np.mean(y_indices))

            # 해당 영역 Depth 추출 및 통계적 정제 (상하위 20% Outlier 제거)
            obj_depths = depth_data[mask]
            valid_depths = obj_depths[obj_depths > 0]
            
            if len(valid_depths) > 10:
                sorted_depths = np.sort(valid_depths)
                n = len(sorted_depths)
                # 상하위 20%를 잘라내어 노이즈(반사, 그림자 등) 차단
                clipped_depths = sorted_depths[int(n*0.2):int(n*0.8)]
                depth_value = float(np.mean(clipped_depths)) if len(clipped_depths) > 0 else float(np.median(valid_depths))
            else:
                depth_value = 0.0
        else:
            # 2. 마스크가 없는 경우 (Fallback)
            cxi, cyi = int(cx), int(cy)
            if 0 <= cyi < height and 0 <= cxi < width:
                window_size = 5
                hw = window_size // 2
                y_start, y_end = max(0, cyi-hw), min(height, cyi+hw+1)
                x_start, x_end = max(0, cxi-hw), min(width, cxi+hw+1)
                roi_depth = depth_data[y_start:y_end, x_start:x_end]
                valid_depths = roi_depth[roi_depth > 0]
                depth_value = float(np.median(valid_depths)) if len(valid_depths) > 0 else 0.0
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
            "center_z": depth_value / 1000.0,
            "mask": mask
        })

    return valid_objects, rgb_image



def get_world_coordinates(target_classes=None, camera_pos=tuple(CAMERA_POS), return_image=False):
    """
    탐지된 객체의 픽셀 좌표와 Depth를 마커 기준의 월드 좌표계로 변환합니다.
    마커가 감지되지 않으면 IMU와 camera_pos를 사용하는 Fallback 모드로 동작합니다.
    return_image=True 설정 시, 시각화 마커와 좌표값이 합성된 이미지가 함께 반환됩니다.
    """
    
    objects, rgb_image = _get_object_pos(target_classes)
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
        
        # 90도 꺾임(Local Minima) 방지를 위해 평면 마커 전용 알고리즘(IPPE_SQUARE) 사용
        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        
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
                aruco.drawDetectedMarkers(rgb_image, corners, None)
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
                p_world_z = mz * Z_SCALING_FACTOR  # 상/하
                
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

            # ============ 4. 이동 평균 필터 적용 (떨림 방지) ============
            cls = obj["class_name"]
            current_pos = (p_world_x, p_world_y, p_world_z)
            
            if cls not in _object_history:
                _object_history[cls] = []
            
            _object_history[cls].append(current_pos)
            if len(_object_history[cls]) > HISTORY_SIZE:
                _object_history[cls].pop(0)
            
            # 히스토리 내 좌표들의 평균값 계산
            avg_coords = np.mean(_object_history[cls], axis=0)
            smooth_x, smooth_y, smooth_z = avg_coords
            
            world_objects.append({
                "class_name": cls,
                "world_x": round(float(smooth_x), 4), 
                "world_y": round(float(smooth_y), 4), 
                "world_z": round(float(smooth_z), 4)  
            })
            
            if return_image:
                w, h = obj["w"], obj["h"]
                # 시각화 시에는 부드러워진 좌표(smooth_*)를 사용하여 텍스트 표시
                x1, y1 = int(cx - w / 2), int(cy - h / 2)
                x2, y2 = int(cx + w / 2), int(cy + h / 2)
                
                # Bounding box
                cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Mask Overlay
                if obj.get("mask") is not None:
                    mask = obj["mask"]
                    overlay = rgb_image.copy()
                    overlay[mask] = (0, 255, 0)
                    cv2.addWeighted(overlay, 0.3, rgb_image, 0.7, 0, rgb_image)

                # Info Text
                text1 = f"{cls}"
                text2 = f"X:{smooth_x:.3f} Y:{smooth_y:.3f} Z:{smooth_z:.3f}"
                # ... 텍스트 출력 로직 ...
                cv2.putText(rgb_image, text1, (x1, max(y1 - 22, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(rgb_image, text2, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if return_image:
        return world_objects, rgb_image
        
    return world_objects


def get_objects_world_pos(target_classes=None):
    if target_classes is None:
        from .camera import get_current_targets
        target_classes = get_current_targets()
    return get_world_coordinates(target_classes=target_classes)