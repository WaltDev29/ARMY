'''
2026.02.28
Pybullet 제어 Tool을 가진 Agent 개발
'''


import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from app import create_agent
from langchain.messages import HumanMessage

app = FastAPI(title="ARMY Agent Web UI")

agent = create_agent()
config = {
    "configurable": {"thread_id": "user1"}
}

# 템플릿 경로 설정
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "app", "templates")
INDEX_PATH = os.path.join(TEMPLATE_DIR, "index.html")

class ChatRequest(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

from fastapi import Response
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(content=b"", media_type="image/x-icon", status_code=204)

@app.post("/chat")
async def chat(req: ChatRequest):
    # 에이전트에 메시지 전송 및 응답 스트리밍/호출
    # 웹 UI에서는 스트리밍 결과를 한 번에 받아 반환하는 방식으로 우선 구현
    result = agent.invoke({"messages": [HumanMessage(content=req.message)]}, config=config)
    messages = result.get("messages", [])
    
    response_text = "응답을 생성할 수 없습니다."
    if messages:
        # 마지막 메시지가 에이전트의 응답
        response_text = messages[-1].content
        
    return {"reply": response_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)