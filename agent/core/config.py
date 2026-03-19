from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel
import os

BASE_DIR = Path(__file__).resolve().parent.parent.parent # project root
ENV_PATH = ".env.dofbot"

load_dotenv(BASE_DIR / ENV_PATH, override=True)



class Config(BaseModel):
    VISION_URL:str = os.getenv("VISION_URL")
    BOT_URL:str = os.getenv("BOT_URL")
    DOFBOT:bool = str(os.getenv("DOFBOT", "False")).lower() == "true"

config = Config()



if __name__ == "__main__":
    print(config.VISION_URL)