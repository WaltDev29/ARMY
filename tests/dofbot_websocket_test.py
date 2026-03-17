# app.py
import streamlit as st
import socketio
import time
import queue

API_URL = "http://192.168.25.100:5000"

st.set_page_config(
    page_title="Robot UI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================
# Socket Manager
# ==============================
@st.cache_resource
def get_socket_manager():
    sio = socketio.Client(reconnection=True)

    robot_q = queue.Queue()

    @sio.on("robot_state")
    def on_robot(data):
        if robot_q.qsize() > 2:
            robot_q.get_nowait()
        robot_q.put(data)

    return sio, robot_q


sio, robot_q = get_socket_manager()


# ==============================
# Session State
# ==============================
if "server_data" not in st.session_state:
    st.session_state.server_data = {
        "ee": {"x": 0.0, "y": 0.0, "z": 0.0},
        "joints": [0, 0, 0, 0, 0, 0],
    }

if "gripper" not in st.session_state:
    st.session_state.gripper = 90.0

if "torque" not in st.session_state:
    st.session_state.torque = 1


# ==============================
# Socket connect
# ==============================
if not sio.connected:
    try:
        sio.connect(API_URL, transports=["websocket", "polling"], wait_timeout=3)
    except:
        pass


# ==============================
# Sync robot state
# ==============================
if not robot_q.empty():
    while not robot_q.empty():
        latest = robot_q.get_nowait()

    st.session_state.server_data["ee"] = latest["ee"]
    st.session_state.server_data["joints"] = latest["joints"]


srv = st.session_state.server_data
ee = srv["ee"]
joints = srv["joints"]


# ==============================
# Command send functions
# ==============================

def send_pos_command():
    pos = [
        st.session_state.input_arm_x,
        st.session_state.input_arm_y,
        st.session_state.input_arm_z,
    ]

    if sio.connected:
        sio.emit("set_pos", {"pos": pos})


def send_gripper_command():
    val = st.session_state.gripper

    if sio.connected:
        sio.emit("set_gripper", {"gripper": val})


def send_torque_command():
    val = st.session_state.torque

    if sio.connected:
        sio.emit("set_torque", {"torque": val})


# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.header("System Status")

    status_color = "green" if sio.connected else "red"
    status_text = "Connected" if sio.connected else "Disconnected"

    st.markdown(f"Server: :{status_color}[{status_text}]")

    if st.button("Reconnect"):
        sio.disconnect()
        st.rerun()


# ==============================
# Layout
# ==============================

col1, col2 = st.columns([2, 1])


# ==============================
# Column 1
# ==============================
with col1:

    st.subheader("Live Feed")

    st.markdown(
        f'<img src="{API_URL}/" width="100%" style="border-radius:10px;">',
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("Gripper Control")

    st.slider(
        "Gripper Angle",
        10.0,
        170.0,
        key="gripper",
        on_change=send_gripper_command
    )


# ==============================
# Column 2
# ==============================
with col2:

    st.subheader("IK Control")

    st.number_input("Target X", value=0.2, step=0.01, key="input_arm_x")
    st.number_input("Target Y", value=0.0, step=0.01, key="input_arm_y")
    st.number_input("Target Z", value=0.15, step=0.01, key="input_arm_z")

    st.button(
        "Send Position",
        on_click=send_pos_command,
        use_container_width=True
    )

    st.divider()

    st.subheader("Torque")

    st.radio(
        "Servo Torque",
        options=[1, 0],
        format_func=lambda x: "ON" if x == 1 else "OFF",
        key="torque",
        on_change=send_torque_command
    )

    st.divider()

    st.subheader("Robot State")

    st.info(f"EE: ({ee['x']:.3f}, {ee['y']:.3f}, {ee['z']:.3f})")

    with st.expander("Joint Angles"):
        for i, j in enumerate(joints):
            st.write(f"J{i+1}: {j:.2f}°")


# ==============================
# Auto refresh
# ==============================
time.sleep(0.05)
st.rerun()