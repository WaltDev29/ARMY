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
            if results and len(results) > 0:
                res = results[0]
                if hasattr(res, 'masks') and res.masks is not None:
                    import cv2
                    kernel = np.ones((3, 3), np.uint8) # 침식용 커널
                    
                    pred_masks = res.masks.data.cpu().numpy().astype(np.uint8)
                    
                    # FastSAM이 반환한 결과(res.boxes)의 순서가 입력 bboxes와 다를 수 있으므로,
                    # 각 입력 바운딩 박스 중심과의 거리를 비교하여 올바른 마스크를 1:1 매칭합니다.
                    if hasattr(res, 'boxes') and res.boxes is not None:
                        pred_boxes = res.boxes.xyxy.cpu().numpy()
                        
                        for input_box in bboxes:
                            cx1 = (input_box[0] + input_box[2]) / 2
                            cy1 = (input_box[1] + input_box[3]) / 2
                            
                            best_idx = 0
                            min_dist = float('inf')
                            
                            for j, p_box in enumerate(pred_boxes):
                                cx2 = (p_box[0] + p_box[2]) / 2
                                cy2 = (p_box[1] + p_box[3]) / 2
                                dist = (cx1 - cx2)**2 + (cy1 - cy2)**2
                                if dist < min_dist:
                                    min_dist = dist
                                    best_idx = j
                            
                            # 매칭된 마스크 추출 및 침식
                            if best_idx < len(pred_masks):
                                eroded = cv2.erode(pred_masks[best_idx], kernel, iterations=1)
                                masks.append(eroded.astype(bool))
                            else:
                                masks.append(None)
                    else:
                        # Fallback: pred_boxes가 없을 경우 단순 인덱스 접근 (비상용)
                        for i in range(len(bboxes)):
                            if i < len(pred_masks):
                                eroded = cv2.erode(pred_masks[i], kernel, iterations=1)
                                masks.append(eroded.astype(bool))
                            else:
                                masks.append(None)
                else:
                    masks = [None] * len(bboxes)
            else:
                masks = [None] * len(bboxes)

        except Exception as e:
            logger.warning(f"마스크 맵핑 중 오류 발생: {e}")
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
