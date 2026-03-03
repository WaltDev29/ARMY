from langchain.tools import tool
from ...core.config import config
import requests



def func():
    return "this is dummy tool"

func_tool = tool(
    func,
    description="it's dummy tool"
)

tools = [func_tool]