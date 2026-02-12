import cv2
# from picamera2 import Picamera2
from ultralytics import YOLO

import angle_calc

# copy pasted from keypoint acquisition
def get_keypoint_position(keypoint_num, axis='x'):
    """ 
    Keypoint reference:
        0: nose          5: left_shoulder  10: right_wrist    15: left_ankle
        1: left_eye      6: right_shoulder 11: left_hip       16: right_ankle
        2: right_eye     7: left_elbow     12: right_hip
        3: left_ear		 8: right_elbow    13: left_knee
        4: right_ear	 9: left_wrist     14: right_knee
    """
    if not 0 <= keypoint_num <= 16:
        raise ValueError("Keypoint number must be between 0 and 16")
    if axis.lower() not in ['x', 'y']:
        raise ValueError("Axis must be 'x' or 'y'")
    
    # Get the keypoint data
    keypoint = results[0].keypoints.xyn[0][keypoint_num]
    
    # Return x or y coordinate based on axis parameter
    return keypoint[0].item() if axis.lower() == 'x' else keypoint[1].item()

# data for form correction
form_data = {"bicep curl":{"min":27, "max":175}, "tricep press":{"min":90, "max":165}}

# checking if all 3 keypoints for an arm exist
def check_arm_kp_exist(side):
    left_exist = True
    right_exist = True
    left_arm_kpts = [5, 7, 9]
    right_arm_kpts = [6, 8, 10]
    if side.lower() == 'left':
        for keypoint_num in left_arm_kpts:
            keypoint_visi = results[0].keypoints.data[0][keypoint_num][2]
            if keypoint_visi < 0.8:
                left_exist = False
                #print("left no exist")
                return left_exist
        #print("left yes exist")
        return left_exist
    else:
        for keypoint_num in right_arm_kpts:
            keypoint_visi = results[0].keypoints.data[0][keypoint_num][2]
            if keypoint_visi < 0.8:
                right_exist = False
                #print("right no exist")
                return right_exist
        #print("right yes exist")
        return right_exist

def get_left_arm_kp():
    left_shoulder_x = get_keypoint_position(5, axis = 'x')
    left_shoulder_y = get_keypoint_position(5, axis = 'y')
    #print(f"Left Shoulder: ({left_shoulder_x:.2f}, {left_shoulder_y:.2f})")
    left_elbow_x = get_keypoint_position(7, axis = 'x')
    left_elbow_y = get_keypoint_position(7, axis = 'y')
    #print(f"Left Elbow: ({left_elbow_x:.2f}, {left_elbow_y:.2f})")
    left_wrist_x = get_keypoint_position(9, axis = 'x')
    left_wrist_y = get_keypoint_position(9, axis = 'y')
    #print(f"Left Wrist: ({left_wrist_x:.2f}, {left_wrist_y:.2f})")
    return left_shoulder_x, left_shoulder_y, left_elbow_x, left_elbow_y, left_wrist_x, left_wrist_y
    
def get_right_arm_kp():
    right_shoulder_x = get_keypoint_position(6, axis = 'x')
    right_shoulder_y = get_keypoint_position(6, axis = 'y')
    #print(f"Right Shoulder: ({right_shoulder_x:.2f}, {right_shoulder_y:.2f})")
    right_elbow_x = get_keypoint_position(8, axis = 'x')
    right_elbow_y = get_keypoint_position(8, axis = 'y')
    #print(f"Right Elbow: ({right_elbow_x:.2f}, {right_elbow_y:.2f})")
    right_wrist_x = get_keypoint_position(10, axis = 'x')
    right_wrist_y = get_keypoint_position(10, axis = 'y')
    #print(f"Right Wrist: ({right_wrist_x:.2f}, {right_wrist_y:.2f})")
    return right_shoulder_x, right_shoulder_y, right_elbow_x, right_elbow_y, right_wrist_x, right_wrist_y

# form correction feedbackmaxxer
def form_corrector(form_data, angle, exercise):
    good_text = "You are doing great! Keep it up!"
    for name in form_data.keys():
        if name.lower() == exercise.lower():
            if form_data[exercise]["min"] + 10 < angle < form_data[exercise]["min"] + 20:
                contract_more_text = "Your arm is not contracted enough. Keep going!"
                return contract_more_text
            elif form_data[exercise]["max"] - 20 < angle < form_data[exercise]["max"] - 10:
                extend_more_text = "Your arm is not extended enough. Keep going!"
                return extend_more_text
            else:
                return good_text
        else:
            return "Exercise not in system"

"""
# Set up the camera with Picam
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 1280)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
"""

# Load our YOLO26 model
model = YOLO("yolo26n-pose.pt")
cam = cv2.VideoCapture(0)

while True:
    # Capture a frame from the camera
    ret, frame = cam.read()
    
    # if frame is read correctly ret is True
    if not ret:
        print("Can't recieve frame. Exiting...")
        break

    # Run YOLO model on the captured frame and store the results
    # max_det is the maximum object detection (per class i think). currently a crude method to limit the detection
    results = model.predict(frame, imgsz = 640, max_det = 1)
    
    # isolating the user from all the objects detected (ignore stuff here its lowk not working/might be useless)
    """if results[0].boxes.is_track == True:
        bbox_sizes = {}
        bbox_data = results[0].boxes
        num_bbox = len(bbox_data)
        for row in range(num_bbox):
            if bbox_data[row][6] == 0:
                bbox_size = angle_calc.line_length(bbox_data[row][0],bbox_data[row][1],bbox_data[row][2],bbox_data[row][3])
                bbox_sizes[bbox_data[row][4]] = bbox_size
    biggest_size = max(bbox_sizes.values())
    user_id = bbox_sizes.get(biggest_size)"""

    # Output the visual detection data, we will draw this on our camera preview window
    annotated_frame = results[0].plot()
    
    # Get inference time
    inference_time = results[0].speed['inference']
    fps = 1000 / inference_time  # Convert to milliseconds
    text = f'FPS: {fps:.1f}'

    # Define font and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1,2)[0]
    text_x = annotated_frame.shape[1] - text_size[0] - 10  # 10 pixels from the right
    text_y = text_size[1] + 10  # 10 pixels from the top

    # angle calculation and displaying for both arms
    try:
        left_arm_exist = check_arm_kp_exist('left')
        if left_arm_exist == True:
            ls_x, ls_y, le_x, le_y, lw_x, lw_y = get_left_arm_kp()
            angle_left = angle_calc.angle_calc(ls_x, ls_y, lw_x, lw_y, le_x, le_y)
            angle_l_text = f"Left arm angle: {angle_left:.2f}"
            #print(angle_l_text)
            cv2.putText(annotated_frame, form_corrector(form_data, angle_left, "bicep curl"), (0, 50), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, angle_l_text, (0, 70), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        right_arm_exist = check_arm_kp_exist('right')
        if right_arm_exist == True:
            rs_x, rs_y, re_x, re_y, rw_x, rw_y = get_right_arm_kp()
            angle_right = angle_calc.angle_calc(rs_x, rs_y, rw_x, rw_y, re_x, re_y)
            angle_r_text = f"Right arm angle: {angle_right:.2f}"
            #print(angle_r_text)
            cv2.putText(annotated_frame, form_corrector(form_data, angle_right, "bicep curl"), (0, 90), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, angle_r_text, (0, 110), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    except (IndexError, AttributeError):
        print("No person detected in frame")

    # Draw the text on the annotated frame
    cv2.putText(annotated_frame, text, (0, 15), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow("Camera", annotated_frame)

    # Exit the program if q is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# release capture
cam.release()
# Close all windows
cv2.destroyAllWindows()