from threading import Lock

import angle_calc
import cv2
from flask import Flask, Response, redirect, render_template, request, url_for
from flask_socketio import SocketIO

# Import pose utilities from pose_final.py
from pose_final import (
    check_arm_kp_exist,
    draw_pose,
    exercises_what_kpts,
    form_corrector,
    form_data,
    get_left_kp,
    get_right_kp,
)
from ultralytics import YOLO

app = Flask(__name__)
socketio = SocketIO(
    app, cors_allowed_origins="*", manage_session=False, async_mode="threading"
)

# ── Shared globals (same as pose_demo2.py) ──
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
# use yolo26n-pose.pt for easier development on computer, switch to yolo26n-pose_ncnn_model for better performance on raspberry pi
# model = YOLO("yolo26n-pose.pt")
model = YOLO("yolo26n-pose_ncnn_model")

IMGSZ = 320

selected_id = None

# test exercise variable
exercise_wanted = "ab crunch"

# shared latest outputs
state_lock = Lock()
latest_jpeg = None
latest_pose = {"feedback": "Starting..."}

clicked_point = None   # (x,y) from UI
last_box = None        # (x1,y1,x2,y2) for reacquire

# -- for click selection --
def pick_id_from_click(boxes_xyxy, ids, px, py):
    candidates = []
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        if x1 <= px <= x2 and y1 <= py <= y2:
            area = (x2 - x1) * (y2 - y1)
            candidates.append((area, ids[i], i))
    if not candidates:
        return None, None
    candidates.sort(key=lambda t: t[0])  # smallest area first
    _, sel_id, sel_i = candidates[0]
    return sel_id, sel_i

def iou_tuple(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / float(area_a + area_b - inter + 1e-9)

# ── Core frame processor ──
def process_frame(frame):
    """
    Replicates pose_final.py's while loop logic.

    Returns: (out_frame, pose_json)
    out_frame = annotated OpenCV frame (the `out` variable)
    pose_json = data for Socket.IO
    """
    global selected_id, exercise_wanted, clicked_point, last_box

    results = model.track(frame, imgsz=IMGSZ, classes=[0], persist=True, verbose=False)[
        0
    ]

    out = frame.copy()

    boxes = results.boxes
    kpts = results.keypoints

    # No person detected
    if boxes is None or boxes.id is None or len(boxes) == 0:
        cv2.putText(
            out,
            "No person detected",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        return out, {"feedback": "No person detected"}

    ids = boxes.id.int().cpu().tolist()

    # if no selection yet, pick the biggest person (closest to camera) as default
    # if selected_id is None:
    #     areas = []
    #     xyxy = boxes.xyxy.cpu().tolist()
    #     for i, (x1, y1, x2, y2) in enumerate(xyxy):
    #         area = (x2 - x1) * (y2 - y1)
    #         areas.append((area, ids[i], i))
    #     areas.sort(reverse=True)
    #     selected_id = areas[0][1]
    xyxy = boxes.xyxy.cpu().tolist()
    xyxy = [list(map(int, b)) for b in xyxy]  # int boxes

    # If we have a click, lock to whoever was clicked
    if clicked_point is not None:
        px, py = clicked_point
        sel, sel_i = pick_id_from_click(xyxy, ids, px, py)
        if sel is not None:
            selected_id = sel
            # store last_box for re-acquire
            last_box = tuple(xyxy[sel_i])
        clicked_point = None

    if selected_id is None:
        for (x1, y1, x2, y2) in xyxy:
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(out, "Click a person to select", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        return out, {"feedback": "Click a person to select"}
    
    if selected_id is not None and selected_id not in ids:
        # try reacquire using last_box (best IoU)
        if last_box is not None:
            best_i = None
            best_score = 0.0
            for i, b in enumerate(xyxy):
                score = iou_tuple(b, last_box)
                if score > best_score:
                    best_score = score
                    best_i = i
            if best_i is not None and best_score > 0.10:
                selected_id = ids[best_i]
                last_box = tuple(xyxy[best_i])
            else:
                selected_id = None  # require click again
        else:
            selected_id = None

    pose_json = {}

    # Person selected → run angles + feedback
    if selected_id in ids:
        i = ids.index(selected_id)
        x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().tolist())
        last_box = (x1, y1, x2, y2)

        if kpts is not None:
            kxy = kpts.xy[i].cpu().numpy()
            kconf = (
                kpts.conf[i].cpu().numpy()
                if hasattr(kpts, "conf") and kpts.conf is not None
                else None
            )

            left_idxs, right_idxs = exercises_what_kpts[exercise_wanted]
            left_ok = check_arm_kp_exist(kconf, left_idxs, th=0.8)
            right_ok = check_arm_kp_exist(kconf, right_idxs, th=0.8)
            angle_left, angle_right = None, None

            if left_ok:
                l1x, l1y, l2x, l2y, l3x, l3y = get_left_kp(kxy, exercise_wanted)
                angle_left = angle_calc.angle_calc(l1x, l1y, l3x, l3y, l2x, l2y)

            if right_ok:
                r1x, r1y, r2x, r2y, r3x, r3y = get_right_kp(kxy, exercise_wanted)
                angle_right = angle_calc.angle_calc(r1x, r1y, r3x, r3y, r2x, r2y)

            # Generate JSON to pass to Socket.IO
            pose_json = {
                "exercise": exercise_wanted,
                "left_angle": angle_left,
                "right_angle": angle_right,
                "left_feedback": form_corrector(form_data, angle_left, exercise_wanted)
                if angle_left is not None
                else "Left keypoints not detected",
                "right_feedback": form_corrector(
                    form_data, angle_right, exercise_wanted
                )
                if angle_right is not None
                else "Right keypoints not detected",
            }

            # draw pose overlay
            draw_pose(out, kxy, exercise_wanted, kconf, conf_thres=0.3)

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return out, pose_json  # ← returns annotated frame + JSON

# ── Background camera loop (updates shared latest_jpeg and latest_pose) ──
def camera_loop():
    global latest_jpeg, latest_pose
    while True:
        ret, frame = cap.read()
        if not ret:
            socketio.sleep(0.01)
            continue

        out_frame, pose_json = process_frame(frame)
        ok, buffer = cv2.imencode(".jpg", out_frame)
        if ok:
            with state_lock:
                latest_jpeg = buffer.tobytes()
                latest_pose = pose_json

        # emit pose at ~20–30Hz (tune as needed)
        socketio.emit("pose_data", pose_json)
        socketio.sleep(0.03)


# ── MJPEG stream ──
@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with state_lock:
                jpg = latest_jpeg
            if jpg is None:
                socketio.sleep(0.01)
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            socketio.sleep(0.01)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ── Main page ──
@app.route("/")
def home():
    return redirect(url_for("selection"))


# ── Selection page ──
@app.route("/selection")
def selection():
    return render_template("selection.html")


# ── Workout page (expects query params: pageTitle and video) ──
@app.route("/workout")
def workout():
    pageTitle = request.args.get("pageTitle", "Workout")
    video = request.args.get("video", "")
    return render_template("workout.html", pageTitle=pageTitle, video=video)


# ── Socket.IO handler for exercise selection from frontend ──
@socketio.on("set_exercise")
def set_exercise(data):
    # frontend can send {exercise:"bicep curl"}
    global exercise_wanted
    ex = (data or {}).get("exercise")
    if ex:
        exercise_wanted = ex.lower()


@socketio.on("reset_target")
def reset_target():
    global selected_id
    selected_id = None

@socketio.on("select_point")
def select_point(data):
    global clicked_point, selected_id
    x = int((data or {}).get("x", -1))
    y = int((data or {}).get("y", -1))
    print("CLICK:", x, y)
    if x >= 0 and y >= 0:
        clicked_point = (x, y)
        selected_id = None  # force re-lock on next frame
        last_box = None


if __name__ == "__main__":
    socketio.start_background_task(camera_loop)
    socketio.run(app, host="0.0.0.0", port=3000, debug=False)
