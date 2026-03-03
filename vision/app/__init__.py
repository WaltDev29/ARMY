from fastapi import FastAPI
from pydantic import BaseModel
from .yolo_detect import detect_objects



class Detection(BaseModel):
    objects:list|None



def create_app():
    app = FastAPI()


    @app.get("/detect",
            summary="Yolo Detection",
            description="카메라의 이미지를 YOLO로 추론하여 결과를 반환합니다.",
            response_model=Detection)
    def detect():
        detected = detect_objects()
        print(detected)
        return Detection(objects=detected)


    return app