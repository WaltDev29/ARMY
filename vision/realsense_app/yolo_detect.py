from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# YOLOv8 모델을 전역으로 한 번만 로드하여 속도 최적화 (가장 가벼운 nano 모델 사용)
model = YOLO(BASE_DIR/'yolo11s.pt')

def detect_objects(image):
    """
    이미지를 인자로 받아 해당 이미지에서 오브젝트들을 탐지해,
    각 오브젝트의 클래스명과 xywh(중심x, 중심y, 너비, 높이)를 반환하는 함수
    """
    if image is None:
        return []
        
    # 모델 추론 수행
    results = model(image, verbose=False)
    detections = []
    
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
            
        for box in boxes:
            # xywh 포맷 [center x, center y, width, height]
            x, y, w, h = box.xywh[0].tolist()
            
            # 클래스 ID를 통해 클래스 문자열 확인
            cls_id = int(box.cls[0].item())
            class_name = model.names[cls_id]
            
            detections.append({
                "class_name": class_name,
                "xywh": [x, y, w, h]
            })
            
    return detections
