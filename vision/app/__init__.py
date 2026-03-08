from fastapi import FastAPI, UploadFile, File, status
from pydantic import BaseModel
from .yolo_detect import detect_objects, detect_objects_from_image, realtime_cam
import numpy as np
import cv2


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
    

    @app.post("/detect_from_image",
            summary="전달받은 이미지를 YOLO로 추론하여 결과를 반환합니다.",
            response_model=Detection,
            status_code=status.HTTP_201_CREATED)
    async def detect_from_image(data: UploadFile = File(...)):
        img_bytes = await data.read()

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        detected = detect_objects_from_image(img)
        print(detected)
        return Detection(objects=detected)


    return app


from threading import Thread
t = Thread(target=realtime_cam)
