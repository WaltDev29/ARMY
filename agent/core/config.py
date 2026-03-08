from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel
import os

BASE_DIR = Path(__file__).resolve().parent.parent.parent # project root
ENV_PATH = ".env.pybullet"

load_dotenv(BASE_DIR / ENV_PATH)



class Config(BaseModel):
    VISION_URL:str = os.getenv("VISION_URL")
    BOT_URL:str = os.getenv("BOT_URL")
    DOFBOT:bool = os.getenv("DOFBOT")

config = Config()



if __name__ == "__main__":
    print(config.VISION_URL)