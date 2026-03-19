from langchain_ollama import ChatOllama
from langchain import agents
from .tools.all_tools import tools
from pathlib import Path
from .core.config import config
from langgraph.checkpoint.memory import MemorySaver

BASE_PATH = Path(__file__).resolve().parent

def create_agent():
    llm = ChatOllama(
        model="llama3.1:latest",
        temperature=0.2
    )

    if config.DOFBOT:
        prompt_path = "prompts/system_dofbot.md"
    else:
        prompt_path = "prompts/system_pybullet.md"

    with open(BASE_PATH / prompt_path, "r", encoding="utf-8") as f:
        system_propmt = f.read()

    # MemorySaver를 checkpointer로 등록하여 에이전트 생성
    agent = agents.create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_propmt,
        checkpointer=MemorySaver()
    )

    return agent