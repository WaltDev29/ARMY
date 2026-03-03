from langchain.tools import tool
from pydantic import BaseModel
from ..memory.images import image_store
from ..core.config import config
import requests
import numpy as np
import cv2

def get_image() -> dict|int:
    response = requests.get(url=f"{config.CAMERA_URL}/image")

    if response.status_code != 200: 
        return {"success": False, "code": 500, "message": "이미지 받기 실패"}
    
    img_arr = list(np.frombuffer(response.content, np.uint8))

    img_id = image_store.get_image_key()
    image_store.add_image(img_arr)

    return img_id


get_image_tool = tool( 
    get_image,
    description="get image from camera and save with id"
)



# ============ Image 표시 ============
class ShowImageArgs(BaseModel):
    image_id:int

def show_image(args:ShowImageArgs) -> str|None:
    if args.image_id is None: return "No image id received"


    img_arr = np.array(image_store.store[args.image_id])
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    cv2.imshow(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

show_image_tool = tool(
    show_image,
    description="Shows the image that matches the id. to get id, get_image_tool must precede.",
    args_schema=ShowImageArgs
)



tools = [get_image_tool, show_image_tool]



if __name__ == "__main__":
    img = get_image()

    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    