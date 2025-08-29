import cv2
import time
import os
from datetime import datetime
import mediapipe as mp
import numpy as np

# ---- SETTINGS ---- #
output_dir = "recordings"
os.makedirs(output_dir, exist_ok=True)

CAMERA_INDEX = 1  # 0 = built-in webcam, 1 = Elgato or other external cam
mirror_webcam = True
record_duration_seconds = None  # e.g. 10 to stop after 10s, or None to run until 'q'
save_format = "mp4"
fps = 20

### >>> LIVE POSE TRACKING BOOLEAN <<< ###
live_pose_tracking = True  # <<< SET THIS TO True or False to toggle overlays
### >>> END LIVE TRACKING TOGGLE <<< ###

# ---- POSE SETUP ---- #
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# ---- OUTPUT SETUP ---- #
time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"recording_{time_stamp}.{save_format}"
output_path = os.path.join(output_dir, filename)

cap = cv2.VideoCapture(CAMERA_INDEX)
width, height = int(cap.get(3)), int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v') if save_format == "mp4" else cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

start_time = time.time()
print("\n[INFO] Recording started. Press 'q' to stop.\n")

# ---- ANGLE HELPER ---- #
def calculate_angle(a, b, c):
    a, b, c = map(np.array, [a, b, c])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

# ---- MAIN LOOP ---- #
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if mirror_webcam:
        frame = cv2.flip(frame, 1)

    if live_pose_tracking:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        h, w = frame.shape[:2]

        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark

            def get_point(name):
                lm = landmarks[mp_pose.PoseLandmark[name]]
                return [lm.x * w, lm.y * h]

            # Left Elbow
            l_shoulder = get_point("LEFT_SHOULDER")
            l_elbow = get_point("LEFT_ELBOW")
            l_wrist = get_point("LEFT_WRIST")
            l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
            cv2.putText(frame, f'{int(l_angle)}°', (int(l_elbow[0]), int(l_elbow[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Right Elbow
            r_shoulder = get_point("RIGHT_SHOULDER")
            r_elbow = get_point("RIGHT_ELBOW")
            r_wrist = get_point("RIGHT_WRIST")
            r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            cv2.putText(frame, f'{int(r_angle)}°', (int(r_elbow[0]), int(r_elbow[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Neck Angle
            try:
                nose = get_point("NOSE")
                neck_angle = calculate_angle(l_shoulder, nose, r_shoulder)
                center_x = int((l_shoulder[0] + r_shoulder[0]) / 2)
                center_y = int((l_shoulder[1] + r_shoulder[1]) / 2)
                cv2.putText(frame, f'{int(neck_angle)}°', (center_x, center_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 255), 2)
            except:
                pass

            # Shoulder Angle (torso angle between hips and shoulder)
            try:
                l_hip = get_point("LEFT_HIP")
                r_hip = get_point("RIGHT_HIP")
                hip_center = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2]
                shoulder_center = [(l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2]
                shoulder_angle = calculate_angle(l_hip, shoulder_center, r_hip)
                center_x = int(shoulder_center[0])
                center_y = int(shoulder_center[1])
                cv2.putText(frame, f'{int(shoulder_angle)}°', (center_x, center_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2)
            except:
                pass

    out.write(frame)
    cv2.imshow("Recording (q to quit)", frame)

    if record_duration_seconds and (time.time() - start_time) > record_duration_seconds:
        print("[INFO] Duration complete. Stopping...")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Recording manually stopped.")
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"[INFO] Saved to {output_path}")
