'''
2026.02.28
Pybullet 제어 Tool을 가진 Agent 개발
'''


import os
import requests
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from app import create_agent
from app.core.config import config as app_config
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
from fastapi.responses import StreamingResponse
import json

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(content=b"", media_type="image/x-icon", status_code=204)

@app.get("/config")
async def get_config():
    return {
        "VISION_URL": app_config.VISION_URL,
        "BOT_URL": app_config.BOT_URL
    }

@app.get("/robot/state")
async def get_robot_state_proxy():
    try:
        resp = requests.get(f"{app_config.BOT_URL}/robot/state", timeout=3.0)
        return resp.json()
    except Exception as e:
        return {"error": str(e), "ee": {"x": 0, "y": 0, "z": 0}, "joints": [0,0,0,0,0,0]}

@app.post("/chat_stream")
async def chat_stream(req: ChatRequest):
    async def generate():
        try:
            # LangGraph의 astream_events를 사용하여 토큰 단위 이벤트를 받습니다.
            async for event in agent.astream_events({"messages": [HumanMessage(content=req.message)]}, config=config, version="v2"):
                kind = event["event"]
                name = event.get("name", "")

                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        # LLM이 텍스트(토큰)를 생성 중입니다.
                        yield f"data: {json.dumps({'type': 'llm_chunk', 'content': chunk.content}, ensure_ascii=False)}\n\n"
                        
                elif kind == "on_tool_start":
                    # 도구 호출 시작
                    input_data = event["data"].get("input", {})
                    yield f"data: {json.dumps({'type': 'tool_start', 'name': name, 'args': str(input_data)}, ensure_ascii=False)}\n\n"
                    
                elif kind == "on_tool_end":
                    # 도구 호출 종료
                    output_data = event["data"].get("output", "")
                    yield f"data: {json.dumps({'type': 'tool_end', 'name': name, 'result': str(output_data)}, ensure_ascii=False)}\n\n"
                    
            # 스트리밍 완료
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"
            
    return StreamingResponse(generate(), media_type="text/event-stream")

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