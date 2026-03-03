from app import create_app
from app.yolo_detect import realtime_cam
import uvicorn
import os
from dotenv import load_dotenv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR/".env")

DEBUG = os.getenv("VISION_DEBUG").lower() == "true"



app = create_app()


if DEBUG:
    from threading import Thread
    t = Thread(target=realtime_cam)

if __name__ == "__main__":
    if DEBUG:
        t.start()

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000
    )