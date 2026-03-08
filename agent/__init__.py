from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
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


    store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    agent_with_memory = RunnableWithMessageHistory(
        agent,
        get_session_history,
        input_messages_key="messages",
        history_messages_key="history"
    )


    return agent_with_memory