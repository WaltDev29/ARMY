import uvicorn
import os
from dotenv import load_dotenv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR/".env")

DEBUG = os.getenv("VISION_DEBUG").lower() == "true"



if os.getenv("REALSENSE").lower() == "true":
    import realsense_app as app
else:
    import app

app = app.create_app()


if __name__ == "__main__":
    if DEBUG:
        app.t.start()

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000
    )