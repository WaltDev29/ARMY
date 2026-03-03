from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel
import os

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = ".env.pybullet"

load_dotenv(BASE_DIR / ENV_PATH)



class Config(BaseModel):
    CAMERA_URL:str = os.getenv("CAMERA_URL")
    BOT_URL:str = os.getenv("BOT_URL")

config = Config()



if __name__ == "__main__":
    print(config.BOT_URL)