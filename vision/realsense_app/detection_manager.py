from ultralytics import YOLO, YOLOWorld
from pathlib import Path
import numpy as np
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

class DetectionManager:
    def __init__(self):
        """
        탐지 매니저 초기화: 표준 YOLOv11과 YOLO-World 모델을 로드합니다.
        """
        try:
            # 1. 표준 YOLO 모델 로드 (상시 감시용)
            logger.info("표준 YOLOv11 모델 로딩 중...")
            self.standard_model = YOLO(BASE_DIR / 'models' / 'yolo11s.pt')
            
            # 2. YOLO-World 모델 로드 (Open-Vocabulary용)
            # 모델이 없으면 자동으로 다운로드됩니다. (yolov8s-world.pt 또는 yolov8m-world.pt 사용 가능)
            logger.info("YOLO-World 모델 로딩 중...")
            self.world_model = YOLOWorld(BASE_DIR / 'models' / 'yolov8s-world.pt') 
            
            # 3. 장치 설정 (CUDA 사용 권장)
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"YOLO 모델 장치 설정: {self.device}")
            self.standard_model.to(self.device)
            self.world_model.to(self.device)

            # 기본 클래스 설정 (YOLO-World 실행을 위한 초기화)
            self.current_world_classes = ["object"]
            self.world_model.set_classes(self.current_world_classes)
            
        except Exception as e:
            logger.error(f"모델 로딩 실패: {e}")
            raise

    def detect(self, image, prompt=None, conf_threshold=0.25):
        """
        image: 입력 이미지 (numpy array)
        prompt: 특정 객체를 찾기 위한 텍스트 프롬프트 리스트 (예: ["yellow block", "red cup"])
        conf_threshold: 탐지 임계값
        """
        if image is None:
            return []

        # 프롬프트 유무에 따라 모델 선택
        if prompt:
            # 새로운 프롬프트가 들어오면 YOLO-World 클래스 재설정
            if isinstance(prompt, str):
                prompt = [prompt]
            
            if prompt != self.current_world_classes:
                logger.info(f"YOLO-World 클래스 재설정: {prompt}")
                self.world_model.set_classes(prompt)
                self.current_world_classes = prompt
            
            results = self.world_model.predict(image, conf=conf_threshold, verbose=False, device=self.device)
        else:
            # 프롬프트가 없으면 상시 표준 모델 사용
            results = self.standard_model.predict(image, conf=conf_threshold, verbose=False, device=self.device)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for box in boxes:
                # xywh 포맷 [center x, center y, width, height]
                x, y, w, h = box.xywh[0].tolist()
                cls_id = int(box.cls[0].item())
                class_name = result.names[cls_id]
                conf = float(box.conf[0].item())

                detections.append({
                    "class_name": class_name,
                    "xywh": [x, y, w, h],
                    "confidence": conf,
                    "box_corner": box.xyxy[0].tolist() # Segmentation Prompt용으로 xyxy도 저장
                })
        
        return detections

# 전역 인스턴스 생성 (사용 편의성)
_manager = None

def get_manager():
    global _manager
    if _manager is None:
        _manager = DetectionManager()
    return _manager

def detect_objects(image, prompt=None, conf_threshold=0.25):
    """
    기존 yolo_detect.py와의 호환성을 유지하면서 하이브리드 기능을 제공하는 래퍼 함수
    """
    manager = get_manager()
    return manager.detect(image, prompt, conf_threshold)
