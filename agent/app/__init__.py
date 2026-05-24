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
    import logging
    logger = logging.getLogger("Agent")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
        # 불필요한 외부 라이브러리(HTTP 통신 등) INFO 로그 숨기기
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        
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
        logger.info(f"========== [Node: PLANNER] ==========")
        logger.info(f"[New Plan]:\n{res.content}\n======================================")
        return {"plan": res.content}

    # Excutor
    def excutor(state:MyState):

        # 1. 툴 노드에서 돌아온 경우 (방금 실행된 툴의 결과 로깅)
        last_msg = state["messages"][-1]
        if last_msg.type == "tool":
            logger.info(f"[Tool Return] {last_msg.name} => {last_msg.content}")

        # 2. 현재 상태(노드, 계획) 로깅
        current_plan = state.get("plan", "계획 없음(Plan 미설정)")
        logger.info(f"========== [Node: EXECUTOR] ==========")
        logger.info(f"[Current Plan]: {current_plan}")

        message = state["messages"]
        if system_message:
            message = [system_message] + message
        
        # LLM 추론
        res = llm_with_tools.invoke(message)

        # 3. LLM 출력 로깅
        if res.content:
            logger.info(f"[LLM Output]: {res.content}")

        # 4. 사용할 툴과 파라미터 로깅
        if hasattr(res, 'tool_calls') and res.tool_calls:
            for tc in res.tool_calls:
                logger.info(f"[Tool Call] {tc['name']} | Params: {tc['args']}")
        
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