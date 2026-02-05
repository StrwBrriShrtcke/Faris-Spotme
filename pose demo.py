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

# initialising variables to track previous keypoint data 
left_arm_prev_data = [[0,0],[0,0],[0,0]]
right_arm_prev_data = [[0,0],[0,0],[0,0]]

# checking if all 3 keypoints for an arm exist
"""
ok so this function is supposed to check if all 3 keypoints for an arm are visible on screen
so if they arent visible then dont bother calculating angle cos how the pose estimation works 
is that even though the keypoints are offscreen, the coordinates for them still exist.
but this function doesnt work because anytime you move on screen the keypoint coordinates 
also update and move. i originally thought it would keep its last known coordinates, but apparently thats
not the case.
so i can't figure out how to code this.
ill do more research into it but for now this isnt really working how its supposed to, but it
doesnt impact the rest of the code (i think)
"""
def check_arm_kp_exist(side, l_prev, r_prev):
    left_exist = True
    right_exist = True
    left_arm_kpts = [5, 7, 9]
    right_arm_kpts = [6, 8, 10]
    left_arm_kp_data = [[], [], []]
    right_arm_kp_data = [[], [], []]
    num_same = []
    counter = 0
    if side.lower() == 'left':
        for keypoint_num in left_arm_kpts:
            keypoint_data = results[0].keypoints.xyn[0][keypoint_num]
            left_arm_kp_data[counter].append(keypoint_data[0].item())
            left_arm_kp_data[counter].append(keypoint_data[1].item())
            counter += 1
        #print(f"Left arm kp list: {left_arm_kp_data}")
        for part in range(0,3):
            counter = 0
            for coord in range(0,2):
                if left_arm_kp_data[part][coord] == l_prev[part][coord]:
                    counter += 1
            num_same.append(counter)
        print(f"num_same_l: {num_same}")
        for i in num_same:
            if i == 2:
                left_exist = False
                return left_exist
        return left_exist
    else:
        for keypoint_num in right_arm_kpts:
            keypoint_data = results[0].keypoints.xyn[0][keypoint_num]
            right_arm_kp_data[counter].append(keypoint_data[0].item())
            right_arm_kp_data[counter].append(keypoint_data[1].item())
            counter += 1
        for part in range(0,3):
            counter = 0
            for coord in range(0,2):
                if right_arm_kp_data[part][coord] == r_prev[part][coord]:
                    counter += 1
            num_same.append(counter)
        print(f"num_same_r: {num_same}")
        for i in num_same:
            if i == 2:
                right_exist = False
                return right_exist
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

"""
# Set up the camera with Picam
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 1280)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
"""

# Load our YOLO11 model
model = YOLO("yolo11n-pose.pt")
cam = cv2.VideoCapture(0)

while True:
    # Capture a frame from the camera
    ret, frame = cam.read()
    
    # if frame is read correctly ret is True
    if not ret:
        print("Can't recieve frame. Exiting...")
        break

    # Run YOLO model on the captured frame and store the results
    results = model.predict(frame, imgsz = 640)
    
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
        left_arm_exist = check_arm_kp_exist('left', left_arm_prev_data, right_arm_prev_data)
        if left_arm_exist == True:
            ls_x, ls_y, le_x, le_y, lw_x, lw_y = get_left_arm_kp()
            angle_left = angle_calc.angle_calc(ls_x, ls_y, lw_x, lw_y, le_x, le_y)
            angle_l_text = f"Left arm angle: {angle_left:.2f}"
            #print(angle_l_text)
            cv2.putText(annotated_frame, angle_l_text, (0, 50), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            # updating prev data variable
            left_arm_val = [ls_x, ls_y, le_x, le_y, lw_x, lw_y]
            counter = 0
            for i in range(0,3):
                for n in range(0,2):
                    left_arm_prev_data[i][n] = left_arm_val[counter]
                    counter += 1

        right_arm_exist = check_arm_kp_exist('right', left_arm_prev_data, right_arm_prev_data)
        if right_arm_exist == True:
            rs_x, rs_y, re_x, re_y, rw_x, rw_y = get_right_arm_kp()
            angle_right = angle_calc.angle_calc(rs_x, rs_y, rw_x, rw_y, re_x, re_y)
            angle_r_text = f"Right arm angle: {angle_right:.2f}"
            #print(angle_r_text)
            cv2.putText(annotated_frame, angle_r_text, (0, 70), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            # updating prev arm variable
            right_arm_val = [rs_x, rs_y, re_x, re_y, rw_x, rw_y]
            counter = 0
            for i in range(0, 3):
                for n in range(0, 2):
                    right_arm_prev_data[i][n] = right_arm_val[counter]
                    counter += 1
    
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