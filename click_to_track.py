import cv2
import numpy as np
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
model = YOLO("yolo26n-pose.pt")

# ---------------- Pose drawing ----------------
SKELETON = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),  # arms (your subset)
]


def draw_pose(out, kpts_xy, kpts_conf=None, conf_thres=0.3):
    for a, b in SKELETON:
        xa, ya = kpts_xy[a]
        xb, yb = kpts_xy[b]
        if kpts_conf is not None and (
            kpts_conf[a] < conf_thres or kpts_conf[b] < conf_thres
        ):
            continue
        if xa > 0 and ya > 0 and xb > 0 and yb > 0:
            cv2.line(out, (int(xa), int(ya)), (int(xb), int(yb)), (0, 255, 0), 2)

    for idx, (x, y) in enumerate(kpts_xy):
        if x <= 0 or y <= 0:
            continue
        if kpts_conf is not None and kpts_conf[idx] < conf_thres:
            continue
        cv2.circle(out, (int(x), int(y)), 4, (0, 255, 0), -1)


# ---------------- Click-to-select ----------------
selected_id = None
last_results = None
clicked_point = None  # (x, y)


def point_in_box(px, py, x1, y1, x2, y2):
    return (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)


def choose_box_from_click(results, px, py):
    """
    Pick the *best* box under click.
    If multiple boxes contain the click, choose the smallest area (most specific).
    Returns selected_id (int) or None.
    """
    r = results[0]
    boxes = r.boxes
    if boxes is None or boxes.id is None:
        return None

    ids = boxes.id.int().cpu().tolist()
    xyxy = boxes.xyxy.cpu().tolist()

    candidates = []
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        if point_in_box(px, py, x1, y1, x2, y2):
            area = (x2 - x1) * (y2 - y1)
            candidates.append((area, ids[i]))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0])  # smallest area first
    return int(candidates[0][1])


def on_mouse(event, x, y, flags, param):
    global selected_id, clicked_point, last_results
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        if last_results is not None:
            picked = choose_box_from_click(last_results, x, y)
            if picked is not None:
                selected_id = picked
                print("Selected target ID:", selected_id)
            else:
                print("Clicked, but no person box under cursor.")


cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera", on_mouse)

# ---------------- Main loop ----------------
while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Track only persons, keep IDs persistent
    results = model.track(frame, imgsz=320, classes=[0], persist=True, verbose=False)
    last_results = results
    r = results[0]

    blurred = cv2.GaussianBlur(frame, (31, 31), 0)
    out = blurred.copy()

    boxes = r.boxes
    kpts = r.keypoints

    # If nothing detected, just show blurred frame
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
        cv2.imshow("Camera", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            selected_id = None
        continue

    ids = boxes.id.int().cpu().tolist()
    xyxy = boxes.xyxy.cpu().tolist()

    # If not selected yet, draw all boxes to help user click
    if selected_id is None:
        cv2.putText(
            out,
            "Click a person to select. Press r to reset.",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(
                out,
                f"ID {ids[i]}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        cv2.imshow("Camera", out)
    else:
        # Selected: render only that target
        if selected_id in ids:
            i = ids.index(selected_id)
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().tolist())

            # Keep target region sharp
            out[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

            # Draw pose for target only
            if kpts is not None:
                kxy = kpts.xy[i].cpu().numpy()
                kconf = None
                if hasattr(kpts, "conf") and kpts.conf is not None:
                    kconf = kpts.conf[i].cpu().numpy()
                draw_pose(out, kxy, kconf, conf_thres=0.3)

            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                out,
                f"TRACKING ID {selected_id}  (r: reset)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
        else:
            # Target not in frame
            cv2.putText(
                out,
                "Target lost. Press r then click again.",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )

        # FPS
        inf = r.speed.get("inference", None)
        if inf:
            fps = 1000.0 / inf
            cv2.putText(
                out,
                f"FPS: {fps:.1f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

        cv2.imshow("Camera", out)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("r"):
        selected_id = None
        print("Selection reset. Click to select again.")

cap.release()
cv2.destroyAllWindows()
