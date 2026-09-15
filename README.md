# ARMY Agent (Agentic Robot Manipulation System)

본 프로젝트는 대형언어모델(LLM)과 Vision AI 모델을 결합하여, 5축 다관절 로봇(DOFBot)을 자연어로 자율 제어하는 **VLA(Vision-Language-Action) 기반의 지능형 로봇 제어 시스템**입니다.  
물리 하드웨어(DOFBot)뿐만 아니라 **PyBullet 기반 가상 시뮬레이션 환경**을 지원하여 별도의 로봇 하드웨어 없이도 동작을 테스트하고 검증할 수 있습니다.

<br>

---

## 🏗️ 시스템 아키텍처 (System Architecture)

본 시스템은 3개의 독립된 마이크로서비스로 구성되어 있습니다.

```mermaid
flowchart TD
    User([사용자 / Web UI]) <-->|HTTP / SSE / WebSocket| Agent[Agent Server\n:8000\nLangGraph, OpenAI]
    Agent <-->|REST API\n3D 좌표 / 마스크 요청| Vision[Vision Server\n:8001\nYOLOv11, YOLO-World, FastSAM]
    Agent <-->|SocketIO / REST\n관절 & 그리퍼 제어| Robot[Robot Server\n:5000\nPyBullet Sim or DOFBot HW]
    Vision -.->|RealSense / Camera| RealWorld[실제 환경 / 카메라]
    Robot -.->|관절 제어| HW[실제 Dofbot or PyBullet GUI]
```

- **[AI 에이전트 모듈 (`agent/`)](file:///d:/project/ARMY_agent/agent)** (Port `8000`): LangGraph 기반 에이전트 서버. 사용자 자연어 명령을 분석하고 적절한 도구(비전 탐색, 로봇 모션 제어)를 호출하여 피드백 제공.
- **[로봇 제어 모듈 (`robot/`)](file:///d:/project/ARMY_agent/robot)** (Port `5000`): PyBullet 가상 시뮬레이터 또는 물리 Dofbot 하드웨어와 통신하는 제어 서버 (Flask/SocketIO).
- **[비전 모듈 (`vision/`)](file:///d:/project/ARMY_agent/vision)** (Port `8001`): YOLOv11(상시 탐지), YOLO-World(Open-Vocabulary 탐색), FastSAM(세그멘테이션) 및 Depth 기반 3D 좌표 변환 서버 (FastAPI).

<br>

---

## 🛠️ 주요 도구 및 기술 스택 (Tech Stack)

- **AI / LLM:** LangGraph, LangChain, OpenAI GPT-4o / GPT-4o-mini, Ollama (Gemma, MiniMax 등 로컬 LLM)
- **Vision AI:** Ultralytics YOLOv11, YOLO-World, FastSAM
- **Simulation & Hardware:** PyBullet Simulation, DOFBot (5-DOF Robotic Arm), Intel RealSense D435
- **Backend & Comm:** FastAPI, Flask, SocketIO, Uvicorn, Python 3.10+
- **Frontend:** HTML5, Vanilla JS, CSS (Glassmorphism), Web Speech API (STT/TTS)

<br>

---

## ⚙️ 환경 설정 (Environment Setup)

프로젝트 루트 및 각 모듈 디렉토리에 제공되는 예시 파일(`.env.example`)을 참고하여 환경변수 파일(`.env`)을 생성합니다.

### 1. 루트 환경변수 설정 (`.env`)
환경에 맞춰 예시 파일을 `.env`로 복사한 후 필요한 값(API 키, 서버 주소 등)을 설정합니다.

- **PyBullet 시뮬레이션 환경 사용 시 (기본 권장):**
  ```bash
  # Windows PowerShell
  Copy-Item .env.pybullet.example .env
  
  # Linux / macOS
  cp .env.pybullet.example .env
  ```
  `.env` 예시:
  ```env
  DOFBOT=False
  VISION_URL=http://localhost:8001
  BOT_URL=http://localhost:5000
  LLM_MODEL=gpt-4o-mini
  LLM_BASE_URL=
  API_KEY=your_openai_api_key_here
  ```

- **실제 Dofbot 하드웨어 환경 사용 시:**
  ```bash
  # Windows PowerShell
  Copy-Item .env.dofbot.example .env
  
  # Linux / macOS
  cp .env.dofbot.example .env
  ```
  `.env` 예시:
  ```env
  DOFBOT=True
  VISION_URL=http://localhost:8001
  BOT_URL=http://192.168.25.100:5000   # 실제 로봇 보드 IP
  LLM_MODEL=gpt-4o-mini
  LLM_BASE_URL=
  API_KEY=your_openai_api_key_here
  ```

### 2. 비전 모듈 환경변수 설정 (`vision/.env`)
비전 모듈 디렉토리로 이동하여 예시 파일을 복사합니다.
```bash
# vision 디렉토리 내에서
cd vision
Copy-Item .env.example .env   # (Linux/macOS: cp .env.example .env)
cd ..
```
`vision/.env` 설정값:
```env
VISION_DEBUG=False
REALSENSE=False   # PyBullet/웹캠 환경: False, RealSense Depth 카메라 연결 시: True
```

<br>

---

## 🚀 실행 가이드 (Quick Start)

시스템 구동을 위해 3개의 터미널에서 각 서비스를 순서대로 실행합니다.

### 🎮 A. PyBullet 시뮬레이션 모드로 실행 (하드웨어 미보유 시)

1. **터미널 1: Robot 시뮬레이터 서버 구동 (Port 5000)**
   ```bash
   cd robot
   conda activate robot
   python main.py
   ```
   > [!NOTE]
   > PyBullet GUI 창이 열리며 가상 환경에 Dofbot 로봇팔 및 시뮬레이션 환경이 로드됩니다.

2. **터미널 2: Vision 서버 구동 (Port 8001)**
   ```bash
   cd vision
   conda activate vision
   python main.py
   ```

3. **터미널 3: AI Agent 서버 구동 (Port 8000)**
   ```bash
   cd agent
   conda activate agent
   python main.py
   ```

4. **웹 관제 UI 접속**
   - **통합 Agent 대시보드:** [http://localhost:8000](http://localhost:8000)
   - **비전 전용 관제 화면:** [http://localhost:8001](http://localhost:8001)

---

### 🦾 B. 실제 DOFBot 하드웨어 모드로 실행

1. **로봇 본체:** Dofbot 보드에서 로봇 제어 서버 구동
2. **비전 서버:** `vision/.env`에서 `REALSENSE=True` 설정 후 `python vision/main.py` 실행
3. **루트 환경변수:** `.env`에서 `DOFBOT=True` 및 `BOT_URL=http://<로봇_IP>:5000` 설정
4. **에이전트 구동:** `python agent/main.py` 실행 후 [http://localhost:8000](http://localhost:8000) 접속

<br>

---

## 🌟 주요 특징 (Key Features)

1. **자연어 기반 로봇 제어**
   - 사용자의 자연어 명령(예: *"빨간색 블록을 잡아서 상자 안에 넣어줘"*)을 분석하여 순차적인 Action Plan 수립 및 실행.
2. **Open-Vocabulary 3D 비전 인식**
   - **YOLO-World + FastSAM:** 사전 학습되지 않은 텍스트 기반 객체 탐색 및 정밀 마스크 추출.
   - **Depth 변환:** 2D 픽셀 좌표를 로봇 엔드이펙터가 도달할 수 있는 3D 월드 좌표(X, Y, Z)로 변환.
3. **안전 제어 및 충돌 방지**
   - 상공 대기(Approach), 안전 고도 하강, 파지(Grip), 이송 및 안전 릴리즈(Drop) 등의 안전 절차를 시스템 프롬프트 및 도구 레벨에서 강제.
4. **실시간 관제 웹 인터페이스**
   - 실시간 스트리밍, 타겟 인식 마스크 시각화, LLM 에이전트 추론/도구 호출 로그 실시간 스트리밍 지원.