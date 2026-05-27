from pydantic import BaseModel, Field




class WorldPosResult(BaseModel):
    class_name: str = Field(..., description="탐지된 객체의 클래스 이름")
    world_x: float = Field(..., description="탐지된 객체의 월드 좌표 x")
    world_y: float = Field(..., description="탐지된 객체의 월드 좌표 y")
    world_z: float = Field(..., description="탐지된 객체의 월드 좌표 z")


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
    detections: list[WorldPosResult] = Field(..., description="탐지된 객체들의 리스트")