from fastapi import FastAPI
from .camera import get_rgb_image, get_depth_data, start_debug_stream, stop_camera, get_intrinsics
from .yolo_detect import detect_objects

def create_app() -> FastAPI:
    app = FastAPI()

    @app.on_event("shutdown")
    async def shutdown_event():
        # 앱 종료 시 카메라 리소스를 반환합니다.
        stop_camera()

    @app.get("/")
    async def root():
        return {"message": "RealSense Vision API is running."}

    @app.get("/detect")
    async def detect():
        """
        카메라에서 이미지를 받아 depth 데이터와 YOLO 추론 결과를 반환합니다.
        """
        # 1. 카메라를 통해 2D RGB 이미지 및 Depth 데이터 가져오기
        rgb_image = get_rgb_image()
        depth_data = get_depth_data()

        if rgb_image is None or depth_data is None:
            return {"error": "카메라에서 프레임을 가져오지 못했습니다."}

        # 2. YOLO를 통해 오브젝트 탐지
        detected_objects = detect_objects(rgb_image)

        results = []
        # depth_data의 shape는 (height, width) 형태
        height, width = depth_data.shape
        
        for obj in detected_objects:
            x, y, w, h = obj["xywh"]
            # 바운딩 박스 중심점 (cx, cy)
            cx, cy = int(x), int(y)
            
            # 깊이 데이터가 이미지 해상도 범위를 벗어나지 않도록 예외 처리
            if 0 <= cy < height and 0 <= cx < width:
                # depth_data의 단위는 밀리미터(mm)
                depth_value = float(depth_data[cy, cx])
            else:
                depth_value = 0.0
                
            results.append({
                "class_name": obj["class_name"],
                "xywh": [x, y, w, h],
                "distance_mm": depth_value
            })

        return {
            "status": "success",
            "detections": results,
            "intrinsics": get_intrinsics()
        }

    return app