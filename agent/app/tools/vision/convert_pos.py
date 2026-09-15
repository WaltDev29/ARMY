import numpy as np

# ============ 카메라 좌표계 기준 오브젝트 좌표 반환  ============
def _pixel_to_camera(center_x, center_y, center_z, cam_width=None, cam_height=None, fov=60.0, fx=None, fy=None, cx=None, cy=None):
    '''
    카메라 좌표계에서의 객체 위치 반환 (2D 픽셀 + Depth -> 3D 카메라 좌표계)
    '''
    if cam_width is None:
        cam_width = center_x * 2 if center_x > 0 else 640
    if cam_height is None:
        cam_height = center_y * 2 if center_y > 0 else 480

    if fx is None or fy is None:
        fx = cam_width / (2 * np.tan(np.deg2rad(fov / 2)))
        fy = cam_height / (2 * np.tan(np.deg2rad(fov / 2)))
        
    if cx is None or cy is None:
        cx = cam_width / 2.0
        cy = cam_height / 2.0

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
