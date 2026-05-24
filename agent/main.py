'''
2026.02.28
Pybullet 제어 Tool을 가진 Agent 개발
'''


from app import create_agent
from langchain.messages import HumanMessage


agent = create_agent()

config = {
    "configurable": {"thread_id": "user1"}
}

while True:
    msg = input("메시지 입력 : ")

    for chunk in agent.stream({"messages": [HumanMessage(content=msg)]}, config=config, stream_mode="updates"):
        pass