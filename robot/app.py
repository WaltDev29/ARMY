import streamlit as st
import requests
import time


# ====================================
# 변수/함수
# ====================================
API = "http://localhost:5000" # pybullet 서버

joint_limits = [
    [-90.0, 90.0],
    [-55.0, 55.0],
    [-65.0, 65.0],
    [-90.0, 90.0],
    [-90.0, 90.0]
]

gripper_limit = [0.0, 0.06]


# ============ Object State ============
object_types = ["teddy", "duck", "soccerball", "mug"]
if "object_type" not in st.session_state:
    st.session_state.object_type = "teddy"

for ot in object_types:
    key = f"{ot}_created"
    if key not in st.session_state:
        st.session_state[key] = False

current_target = st.session_state.object_type
is_created = st.session_state.get(f"{current_target}_created", False)
if "object_fix" not in st.session_state:
    st.session_state.object_fix = False

object_info = {
    "label": "제거" if is_created else "생성",
    "op": "delete" if is_created else "create"
}



# ============ joint 슬라이더 onChange 함수 ============
def send_joint_command():
    current_joints = [st.session_state[f"joint_{i}"] for i in range(5)]
    body = {"joints": current_joints}

    try:
        requests.post(f"{API}/set_joints", json=body, timeout=0.5)
    except :
        pass
    
# ============ gripper 슬라이더 onChange 함수 ============
def send_gripper_command():
    current_gripper = st.session_state[f"gripper"]
    body = {"gripper": current_gripper}

    try:
        requests.post(f"{API}/set_gripper", json=body, timeout=0.5)
    except :
        pass
    
    
# ============ velocity 슬라이더 onChange 함수 ============
def send_velocity_cmd():
    cur_velocity = st.session_state["max_velocity"]
    body = {"max_velocity": cur_velocity}
    
    try:
        requests.post(f"{API}/max_velocity", json=body, timeout=0.5)
    except:
        pass
    
    
# ============ 오브젝트 버튼 onClick 함수 ============
def send_object_cmd():
    body = {
        "object": st.session_state.object_type,
        "op": object_info["op"],
        "fix": st.session_state.object_fix
    }

    try:
        r = requests.post(f"{API}/set_object", json=body, timeout=0.5)
        st.session_state.obj_msg = f"응답: {r.json()}"
        
        target_key = f"{st.session_state.object_type}_created"
        st.session_state[target_key] = not st.session_state[target_key]
        
    except Exception as e:
        st.session_state.obj_msg = f"전송 실패: {e}"


# ============ 오브젝트 위치 버튼 onClick 함수 ============
def send_object_pos():
    if not is_created: 
        st.session_state.obj_pos_msg = "오브젝트를 생성해주세요."
        return
    
    body = {
        "object": st.session_state.object_type,
        "pos": [object_x, object_y, object_z]
    }

    try:
        r = requests.post(f"{API}/set_object_pos", json=body, timeout=0.5)
        st.session_state.obj_pos_msg = "응답:", r.json()
    except Exception as e:
       st.session_state.obj_pos_msg = f"전송 실패: {e}"

           
           
# ====================================
# UI 구성
# ====================================
st.set_page_config(
    page_title="PyBullet Robot UI",
    layout="wide",   # 화면 전체 폭 사용
    initial_sidebar_state="auto"
)

col_1, col_2, col_3 = st.columns([2, 1, 1]) # 화면 3열로 구성

with col_1:
    # ============ 영상 표시 ============
    st.markdown(
        f"""
        <img src="{API}/" width="640">
        """,
        unsafe_allow_html=True
    )
    
    
    # ============ Joint 슬라이더 ============
    st.subheader("Joint Control")

    # 초기값
    if "joints" not in st.session_state:
        st.session_state.joints = [0.0, 0.0, 0.0, 0.0, 0.0]

    sliders = []
    for i in range(5):
        sliders.append(
            st.slider(
                f"Joint {i+1}",
                joint_limits[i][0],
                joint_limits[i][1],
                st.session_state.joints[i],
                step=0.01,
                key=f"joint_{i}",
                on_change=send_joint_command
            )
        )
    
    # ============ Gripper 슬라이더 ============
    st.subheader("Gripper Control")
    if "gripper" not in st.session_state:
        st.session_state.gripper = 0.0
    st.slider(
                "gripper",
                gripper_limit[0],
                gripper_limit[1],
                st.session_state.gripper,
                step=0.001,
                key="gripper",
                on_change=send_gripper_command
            )
    
    # ============ max velocity 슬라이더 ============
    if "max_velocity" not in st.session_state:
        st.session_state.velocity = 20
        
    st.slider(
        "Max Velocity",
        0.0,
        20.0,
        20.0,
        step=0.1,
        key="max_velocity",
        on_change=send_velocity_cmd
    )
        
        

with col_2:    
    # ============ 목표 좌표 입력 ============
    st.subheader("목표 좌표 입력")
    arm_x = st.number_input("ARM_X", value=0.0, step=0.01, format="%.3f")
    arm_y = st.number_input("ARM_Y", value=0.0, step=0.01, format="%.3f")
    arm_z = st.number_input("ARM_Z", value=0.0, step=0.01, format="%.3f")

    if st.button("목표 좌표 보내기"):
        body = {
            "pos": [arm_x, arm_y, arm_z]
        }

        try:
            r = requests.post(f"{API}/set_pos", json=body, timeout=0.5)
            st.write("응답:", r.json())
        except Exception as e:
            st.error(f"전송 실패: {e}")

            
    # ============ End-Effector 좌표 표시 ============
    st.subheader("End Effector Position (EE)")
    ee_placeholder = st.empty()

              
            
with col_3:
    # ============ 오브젝트 생성 ============
    st.subheader("오브젝트 제어")
    
    # 오브젝트 고정 여부
    object_fix = st.radio(
        "오브젝트 고정",
        (True, False)
    )
    st.session_state.object_fix = object_fix
        
    # 오브젝트 종류 선택
    object_type = st.radio(
        "오브젝트 타입",
        ("teddy", "duck", "soccerball", "mug")
    )
    st.session_state.object_type = object_type
    
    # 오브젝트 생성/제거
    st.button("오브젝트" + object_info["label"], on_click=send_object_cmd)
    object_placeholder = st.empty()
    if "obj_msg" in st.session_state: object_placeholder.write(st.session_state.obj_msg)

    
    # ============ 오브젝트 위치 제어 ============
    object_x = st.number_input("OBJECT_X", value=0.0, step=0.01, format="%.3f")
    object_y = st.number_input("OBJECT_Y", value=0.0, step=0.01, format="%.3f")
    object_z = st.number_input("OBJECT_Z", value=0.0, step=0.01, format="%.3f")

    st.button("위치 변경", on_click=send_object_pos)
    
            
    # ============ 오브젝트 좌표 표시 ============
    st.subheader("Object Position")
    object_pos_placeholder = st.empty()
    if "obj_pos_msg" in st.session_state: object_pos_placeholder.write(st.session_state.obj_pos_msg)



# ====================================
# HTTP 요청 (Polling)
# ====================================
while True:
    # ============ GET Robot State (EE + joints) ============
    try:
        r = requests.get(f"{API}/robot_state", timeout=0.5)
        if r.status_code == 200:
            state = r.json()
            ee_placeholder.json(state.get("ee", {}))
            # optional: could display joints if desired
    except Exception:
        ee_placeholder.write("robot_state 연결 실패")

    # ============ GET Object State ============
    try:
        res = requests.get(f"{API}/object_state", timeout=0.5)
        if res.status_code == 200:
            obj = res.json().get("object", {})
            object_pos_placeholder.json(obj)
            # update internal flags based on server info
            st.session_state.teddy_created = obj.get("exists", False) if st.session_state.object_type == "teddy" else st.session_state.teddy_created
            st.session_state.duck_created = obj.get("exists", False) if st.session_state.object_type == "duck" else st.session_state.duck_created
    except Exception:
        object_pos_placeholder.write("object_state 연결 실패")

    time.sleep(0.1)