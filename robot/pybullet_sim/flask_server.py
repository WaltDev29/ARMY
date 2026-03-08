from flask import Flask, Response, request, jsonify
import cv2
import json
import time
from . import shared_data as shared

app = Flask(__name__)

# ============ Video Generator ============
def gen():
    while True:
        with shared.frame_lock:
            if shared.latest_frame is None:
                time.sleep(0.01)
                continue
            frame_to_send = shared.latest_frame.copy()
        
        _, jpeg = cv2.imencode('.jpg', frame_to_send, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        time.sleep(0.05)



# ====================================
# GET
# ====================================

# ============ GET Video ============
@app.route("/")
def video_feed():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ============ GET Image ============
@app.route("/image")
def snapshot():
    with shared.frame_lock:
        if shared.latest_frame is None:
            return "No frame yet", 503
        frame = shared.latest_frame.copy()
        
    _, jpeg = cv2.imencode(".jpg", frame)
    return Response(jpeg.tobytes(), mimetype="image/jpeg")


# ============ GET Depth ============
@app.route("/depth")
def get_depth():
    with shared.frame_lock:
        depth = shared.latest_frame_depth

    return jsonify(depth.tolist())


# ============ GET Robot State ============
@app.route("/robot_state")
def get_robot_state():
    with shared.state_lock:
        ee = shared.robot_state.copy()
        joints = shared.joints_degrees[:]
    return jsonify({"ee": ee, "joints": joints})


# ============ GET Object state ============
@app.route("/object_state")
def get_object_state():
    with shared.state_lock:
        info = shared.object_info.copy()
    return jsonify({"object": info})



# ====================================
# POST
# ====================================

# ============ POST 목표 좌표 ============
@app.route("/set_pos", methods=["POST"])
def set_ee():
    data = request.get_json()
    if "pos" not in data:
        return jsonify({"ok": False, "msg": "잘못된 데이터 형식입니다."}), 400
    
    with shared.cmd_lock:
        shared.command["target_pos"] = data["pos"]
    return jsonify({"ok": True})


# ============ POST Joints 각도 ============
@app.route("/set_joints", methods=["POST"])
def set_joints():
    data = request.get_json()
    if "joints" not in data:
        return jsonify({"ok": False, "msg": "잘못된 데이터"}), 400
        
    with shared.cmd_lock:
        shared.command["joint_cmd"] = data["joints"]
        
    return jsonify({"ok": True})


# ============ POST Gripper ============
@app.route("/set_gripper", methods=["POST"])
def set_gripper():
    data = request.get_json()
    if "gripper" not in data:
        return jsonify({"ok": False, "msg": "잘못된 데이터"}), 400
        
    with shared.cmd_lock:
        shared.command["gripper_cmd"] = data["gripper"]
        
    return jsonify({"ok": True})


# ============ POST 로봇 힘 ============
@app.route("/force", methods=["POST"])
def set_force():
    data = request.get_json()
    if "force" not in data: return jsonify({"ok": False}), 400
    
    with shared.cmd_lock:
        shared.command["force"] = data["force"]
    return jsonify({"ok": True})


# ============ POST Robot 최대 속도 ============
@app.route("/max_velocity", methods=["POST"])
def set_max_velocity():
    data = request.get_json()
    if "max_velocity" not in data: return jsonify({"ok": False}), 400

    with shared.cmd_lock:
        shared.command["max_velocity"] = data["max_velocity"]
    return jsonify({"ok": True})


# ============ POST 오브젝트 생성/삭제 ============
@app.route("/set_object", methods=["POST"])
def set_object():
    data = request.json
    with shared.cmd_lock:
        shared.command["object_cmd"] = data
    return {"ok": True}


# ============ POST 오브젝트 위치 제어 ============
@app.route("/set_object_pos", methods=["POST"])
def set_object_pos():
    data = request.json
    if "pos" not in data: return jsonify({"ok": False}), 400
    
    with shared.state_lock:
        exists = shared.object_info["exists"]
    
    if not exists: return "오브젝트를 먼저 생성해주세요.", 400
    
    with shared.cmd_lock:
        shared.command["object_pos_cmd"] = data["pos"]
    return {"ok": True}



# ====================================
# app 실행
# ====================================
def run_flask():
    print(">>> Flask Server Started on port 5000")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)