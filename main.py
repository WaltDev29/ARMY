'''
2026.02.28
Pybullet 제어 Tool을 가진 Agent 개발
'''


from agent import create_agent
from langchain.messages import AIMessage

agent = create_agent()

config = {
    "configurable": {"session_id": "user1"}
}

while True:
    topic = input("입력 : ")
    if not topic: continue
    if topic.strip().lower() == 'q': break

    message = {"messages": [{"role": "user", "content": topic}]}
    
    response = agent.invoke(message, config=config)
    # ai_messages = [m for m in response["messages"] if isinstance(m, AIMessage)]

    # for ai_objs in ai_messages:
    #     print("\n\n============ AI Response ============")
    #     for ai_attr in ai_objs:
    #         print(f"{ai_attr[0]} : {ai_attr[1]}")

    # Stream 방식 출력
    for chunk in agent.stream(message, config=config):
        print(chunk)
