from pydantic import BaseModel, Field


class DetectionResult(BaseModel):
    class_name: str = Field(..., description="탐지된 객체의 클래스 이름")
    xywh: list[float] = Field(..., description="탐지된 객체의 중앙값과 너비, 높이 (x, y, width, height)")
    distance_mm: float = Field(..., description="탐지된 객체까지의 거리 (밀리미터)")


class Intrinsics(BaseModel):
        width: int = Field(..., description="이미지 해상도. width")
        height: int = Field(..., description="이미지 해상도. height")
        ppx: float = Field(..., description="카메라 중심점(카메라가 바라보는 정중앙). x축 중심")
        ppy: float = Field(..., description="카메라 중심점(카메라가 바라보는 정중앙). y축 중심")
        fx: float = Field(..., description="초점거리(카메라가 얼마나 확대해서 보는지). x축 방향 초점 거리")
        fy: float = Field(..., description="초점거리(카메라가 얼마나 확대해서 보는지). y축 방향 초점 거리")
        pitch: float = Field(..., description="카메라의 pitch 각도 (degrees)")
        roll: float = Field(..., description="카메라의 roll 각도 (degrees)")


class ResponseBase(BaseModel):
    status: str = Field(..., description="응답 상태 (예: 'success' 또는 'error')")
    detections: list[DetectionResult] = Field(..., description="탐지된 객체들의 리스트")

