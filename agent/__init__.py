from langchain_ollama import ChatOllama
from langchain import agents
from .tools.all_tools import tools
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = "prompts/system.md"

def create_agent():
    llm = ChatOllama(
        model="llama3.1:latest",
        temperature=0.2
    )

    with open(BASE_PATH / SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        system_propmt = f.read()


    agent = agents.create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_propmt
    )

    return agent