'''
2026.02.28
Pybullet 제어 Tool을 가진 Agent 개발
'''


from agent import create_agent
from langchain.messages import HumanMessage


# ============ 출력용 파싱 함수 ============
def pretty_print(chunk):
    # 📌 Planner
    if "planner" in chunk:
        plan = chunk["planner"].get("plan", "")
        print(f"\n📌 Plan: {plan}")

    # 🤖 Executor
    if "excutor" in chunk:
        messages = chunk["excutor"].get("messages", [])
        if not messages:
            return 
        
        msg = messages[-1]

        # content 출력
        print(f"\n🤖 Execute: {msg.content}")

        # tool_calls 출력
        tool_calls = getattr(msg, "tool_calls", [])
        if tool_calls:
            print("    🔧 Tool Calls:")
            for tc in tool_calls:
                name = tc.get("name")
                args = tc.get("args")

                # args 이상하게 깨지는 경우 대비
                try:
                    print(f"       - {name}({args})")
                except:
                    print(f"       - {name}(args 파싱 실패)")

    # 🛠 Tools 실행 결과
    if "tools" in chunk:
        messages = chunk["tools"].get("messages", [])
        for msg in messages:
            print(f"🛠 Tool Result [{msg.name}]: {msg.content}")



agent = create_agent()

config = {
    "configurable": {"thread_id": "user1"}
}

while True:
    msg = input("메시지 입력 : ")

    for chunk in agent.stream({"messages": [HumanMessage(content=msg)]}, config=config, stream_mode="updates"):
        pretty_print(chunk)