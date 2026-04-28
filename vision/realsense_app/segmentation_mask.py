from ultralytics import FastSAM
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SegmentationManager:
    def __init__(self, model_path='FastSAM-s.pt'):
        """
        FastSAM 모델을 초기화합니다.
        """
        try:
            logger.info(f"FastSAM 모델 로딩 중 ({model_path})...")
            self.model = FastSAM(model_path)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"FastSAM 장치 설정: {self.device}")
        except Exception as e:
            logger.error(f"FastSAM 로딩 실패: {e}")
            raise

    def get_masks(self, image, bboxes):
        """
        이미지와 바운딩 박스들을 입력받아 각 박스에 대응하는 바이너리 마스크 리스트를 반환합니다.
        """
        if image is None or not bboxes:
            return []

        masks = []
        try:
            # FastSAM 및 YOLO 모델과의 Device Mismatch를 방지하기 위해 텐서로 변환
            tensor_bboxes = torch.tensor(bboxes, device=self.device)
            # 최신 Ultralytics FastSAM API는 predict 시 bboxes 프롬프트를 지원합니다.
            results = self.model.predict(
                source=image, 
                bboxes=tensor_bboxes, # 바운딩 박스 프롬프트 전달
                device=self.device, 
                retina_masks=True, 
                imgsz=640, 
                conf=0.4, 
                verbose=False
            )

            # 결과 리스트에서 각 박스에 대응하는 마스크 추출
            if results:
                import cv2
                kernel = np.ones((3, 3), np.uint8) # 침식용 커널 (크기가 클수록 더 많이 깎임)

                for res in results:
                    if hasattr(res, 'masks') and res.masks is not None:
                        # res.masks.data는 [N, H, W] 형태
                        all_masks_data = res.masks.data.cpu().numpy().astype(np.uint8)
                        for i in range(len(all_masks_data)):
                            # 마스크 침식(Erosion): 경계선의 불안정한 떨림 제거
                            eroded = cv2.erode(all_masks_data[i], kernel, iterations=1)
                            masks.append(eroded.astype(bool))
                    else:
                        masks.extend([None] * len(bboxes))
            
            # 반환 리스트 크기 최종 확인 및 보정
            if len(masks) > len(bboxes):
                masks = masks[:len(bboxes)]
            while len(masks) < len(bboxes):
                masks.append(None)

        except Exception as e:
            logger.warning(f"마스크 추출 중 오류 발생: {e}")
            masks = [None] * len(bboxes)

        return masks

# 싱글톤 인스턴스
_seg_manager = None

def get_segmentation_manager():
    global _seg_manager
    if _seg_manager is None:
        from pathlib import Path
        BASE_DIR = Path(__file__).resolve().parent.parent
        _seg_manager = SegmentationManager(model_path=str(BASE_DIR / 'models' / 'FastSAM-s.pt'))
    return _seg_manager

def generate_masks(image, bboxes):
    """
    이미지와 바운딩 박스 리스트를 받아 마스크 리스트를 반환하는 편의 함수
    """
    manager = get_segmentation_manager()
    return manager.get_masks(image, bboxes)
