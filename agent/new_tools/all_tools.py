from .robot import pybullet_tool
from .robot import dofbot_tool
from .vision import yolo_tool
from .vision import realsense_tool
from ..core.config import config

tools = []

if config.DOFBOT:
    tools.extend(dofbot_tool.tools)
    # tools.extend(realsense_tool.tools)
else:
    tools.extend(pybullet_tool.tools)
    # tools.extend(yolo_tool.tools)