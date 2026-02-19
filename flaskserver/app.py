from flask import Flask, Response, render_template
from flask_socketio import SocketIO
import cv2
import numpy as np
from ultralytics import YOLO
import angle_calc

# ── Import everything from teammate's file ──────────────────────
from pose_demo2 import (
    get_left_arm_kp, get_right_arm_kp,
    check_arm_kp_exist, form_corrector,
    draw_pose, form_data, SKELETON
    )

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# ── Shared globals (same as pose_demo2.py) ───────────────────────
cap = cv2.VideoCapture(0)
model = YOLO("yolo26n-pose.pt")
selected_id = None
last_results = None

# ── Core frame processor (extracted from pose_demo2.py loop) ────
def process_frame(frame):
    """
    Replicates pose_demo2.py's while loop logic.
    Returns: (out_frame, pose_json)
    out_frame = annotated OpenCV frame (the `out` variable)
    pose_json = data for Socket.IO
    """
    global selected_id, last_results

    results = model.track(frame, imgsz=320, classes=[0], persist=True, verbose=False)
    last_results = results
    results_filtered = results[0]

    # teammate's blurred background effect
    blurred = cv2.GaussianBlur(frame, (31, 31), 0)
    out = blurred.copy()  # ← THIS is the `out` from pose_demo2.py

    boxes = results_filtered.boxes
    kpts = results_filtered.keypoints

    pose_json = {"left_feedback": "", "right_feedback": "",
                "left_angle": 0, "right_angle": 0}

    # No person detected
    if boxes is None or boxes.id is None or len(boxes) == 0:
        cv2.putText(out, "No person detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return out, {"feedback": "No person detected"}

    ids = boxes.id.int().cpu().tolist()
    xyxy = boxes.xyxy.cpu().tolist()

    # No person selected yet
    if selected_id is None:
        cv2.putText(out, "No person selected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(out, f"ID {ids[i]}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return out, {"feedback": "Select a person"}

    # Person selected → run angles + feedback (teammate's logic)
    if selected_id in ids:
        i = ids.index(selected_id)
        x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().tolist())
        out[y1:y2, x1:x2] = frame[y1:y2, x1:x2]  # sharp target

        if kpts is not None:
            kxy = kpts.xy[i].cpu().numpy()
            kconf = kpts.conf[i].cpu().numpy() if hasattr(kpts, "conf") and kpts.conf is not None else None

            # Left arm
            if check_arm_kp_exist(kconf, [5, 7, 9], th=0.8):
                ls_x, ls_y, le_x, le_y, lw_x, lw_y = get_left_arm_kp(kxy)
                angle_left = angle_calc.angle_calc(ls_x, ls_y, lw_x, lw_y, le_x, le_y)
                left_fb = form_corrector(form_data, angle_left, "bicep curl")
                cv2.putText(out, left_fb, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                cv2.putText(out, f"Left elbow: {angle_left:.1f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                pose_json["left_angle"] = float(angle_left)
                pose_json["left_feedback"] = left_fb

            # Right arm
            if check_arm_kp_exist(kconf, [6, 8, 10], th=0.8):
                rs_x, rs_y, re_x, re_y, rw_x, rw_y = get_right_arm_kp(kxy)
                angle_right = angle_calc.angle_calc(rs_x, rs_y, rw_x, rw_y, re_x, re_y)
                right_fb = form_corrector(form_data, angle_right, "bicep curl")
                cv2.putText(out, right_fb, (0, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                cv2.putText(out, f"Right elbow: {angle_right:.1f}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                pose_json["right_angle"] = float(angle_right)
                pose_json["right_feedback"] = right_fb

            draw_pose(out, kxy, kconf, conf_thres=0.3)  # teammate's skeleton

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return out, pose_json  # ← returns annotated frame + JSON

# ── MJPEG stream (replaces cv2.imshow) ──────────────────────────
@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            out_frame, _ = process_frame(frame)  # ← get `out`
            _, buffer = cv2.imencode('.jpg', out_frame)
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ── Socket.IO pose data ──────────────────────────────────────────
def pose_loop():
    while True:
        ret, frame = cap.read()
        if ret:
            _, pose_json = process_frame(frame)  # ← get JSON
            with app.app_context():
                socketio.emit('pose_data', pose_json)
        socketio.sleep(0.03)

# ── Main page ────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def connect():
    print("Frontend connected!")

if __name__ == '__main__':
    socketio.start_background_task(pose_loop)
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
