from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage
from .tools.all_tools import tools
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, TypedDict
from .core.config import config


def _load_system_prompt() -> str:
    prompt_path = Path(__file__).resolve().parent / "prompts" / "system_dofbot.md"
    if not prompt_path.exists():
        return ""
    return prompt_path.read_text(encoding="utf-8")


def create_agent():
    # ============ State 정의 ============
    class MyState(TypedDict):
        messages: Annotated[list, add_messages]
        plan: str

    # ============ LLM 정의 ============
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        base_url=config.LLM_BASE_URL,
        api_key="",
        default_headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    # ============ Memory 정의 ============
    memory = MemorySaver()

    # ============ 시스템 프롬프트 로드 ============
    system_prompt_text = _load_system_prompt()
    system_message = SystemMessage(content=system_prompt_text) if system_prompt_text else None

    # ============ Tool 등록 ============
    llm_with_tools = llm.bind_tools(tools)


    # ============ Node 정의 ============
    # Planner
    def planner(state:MyState):
        prompt = f"""
            사용자의 질문을 해결하기 위한 단계별 계획을 세워라.
            질문 : {state["messages"][-1].content}
        """

        res = llm.invoke(prompt)  # tool 없는 순수 LLM (계획만 생성)
        print("planner : ", res.content, "\n")
        return {"plan": res.content}

    # Excutor
    def excutor(state:MyState):
        message = state["messages"]
        if system_message:
            message = [system_message] + message
        res = llm_with_tools.invoke(message)
        print("excutor : ", res.content, "\n")
        return {"messages": [res]}

    # Tool Node
    tool_node = ToolNode(tools)



    # ============ Builder 정의 ============
    builder = StateGraph(MyState)

    # ============ Node 등록 ============
    # builder.add_node("planner", planner)
    builder.add_node("excutor", excutor)
    builder.add_node("tools", tool_node)

    # ============ Node 연결 ============
    builder.add_edge(START, "excutor")

    # ============ 분기점 설정 ============
    builder.add_conditional_edges(
        "excutor",
        tools_condition
    )

    # ============ Loop 연결 ============
    builder.add_edge("tools", "excutor")

    # ============ Builder Compile ============
    graph = builder.compile(checkpointer=memory)


    return graph