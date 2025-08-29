import cv2
import mediapipe as mp
import numpy as np

# ---- SETTINGS ---- #
input_path = "recordings/DRAAGON.mp4"  # Or 0 for webcam
output_path = "recordings/pose_output.mp4"

mirror_input = True           # Set to False if no mirroring needed
save_output = True            # Set to False to disable video recording
show_hand_names = True        # Show both index + name on hand joints

# ---- INIT ---- #
cap = cv2.VideoCapture(input_path)

if save_output:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0,
                          (int(cap.get(3)), int(cap.get(4))))

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# ---- HELPERS ---- #
def calculate_angle(a, b, c):
    a, b, c = map(np.array, [a, b, c])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

paused = False

# ---- MAIN LOOP ---- #
while cap.isOpened():
    if not paused:
        success, frame = cap.read()
        if not success:
            break

        if mirror_input:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb)
        h, w = frame.shape[:2]

        # ---- POSE ---- #
        if pose_results.pose_landmarks:
            mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmarks = pose_results.pose_landmarks.landmark

            for idx, lm in enumerate(landmarks):
                x, y = int(lm.x * w), int(lm.y * h)
                joint_name = mp_pose.PoseLandmark(idx).name
                cv2.putText(frame, f"{idx}:{joint_name}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

            def get_point(name):
                lm = landmarks[mp_pose.PoseLandmark[name]]
                return [lm.x * w, lm.y * h]

            # Left Elbow Angle
            l_shoulder = get_point("LEFT_SHOULDER")
            l_elbow = get_point("LEFT_ELBOW")
            l_wrist = get_point("LEFT_WRIST")
            l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
            cv2.putText(frame, f'{int(l_angle)}°', (int(l_elbow[0]), int(l_elbow[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # Right Elbow Angle
            r_shoulder = get_point("RIGHT_SHOULDER")
            r_elbow = get_point("RIGHT_ELBOW")
            r_wrist = get_point("RIGHT_WRIST")
            r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            cv2.putText(frame, f'{int(r_angle)}°', (int(r_elbow[0]), int(r_elbow[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # Left Wrist Angle (between elbow-wrist-index)
            try:
                l_index = get_point("LEFT_INDEX")
                l_wrist_angle = calculate_angle(l_elbow, l_wrist, l_index)
                cv2.putText(frame, f'{int(l_wrist_angle)}°', (int(l_wrist[0]), int(l_wrist[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
            except:
                pass

            # Right Wrist Angle (between elbow-wrist-index)
            try:
                r_index = get_point("RIGHT_INDEX")
                r_wrist_angle = calculate_angle(r_elbow, r_wrist, r_index)
                cv2.putText(frame, f'{int(r_wrist_angle)}°', (int(r_wrist[0]), int(r_wrist[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
            except:
                pass

            # Neck Angle (LEFT_SHOULDER - NOSE - RIGHT_SHOULDER)
            try:
                nose = get_point("NOSE")
                neck_angle = calculate_angle(l_shoulder, nose, r_shoulder)
                center_x = int((l_shoulder[0] + r_shoulder[0]) / 2)
                center_y = int((l_shoulder[1] + r_shoulder[1]) / 2)
                cv2.putText(frame, f'{int(neck_angle)}°', (center_x, center_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 100, 255), 2, cv2.LINE_AA)
            except:
                pass

            # Torso skeleton overlay (virtual ribs and spine)
            try:
                l_hip = get_point("LEFT_HIP")
                r_hip = get_point("RIGHT_HIP")

                # Draw lines from shoulders to hips (spine sides)
                cv2.line(frame, tuple(map(int, l_shoulder)), tuple(map(int, l_hip)), (100, 200, 255), 2)
                cv2.line(frame, tuple(map(int, r_shoulder)), tuple(map(int, r_hip)), (100, 200, 255), 2)

                # Draw hip bridge
                cv2.line(frame, tuple(map(int, l_hip)), tuple(map(int, r_hip)), (0, 200, 200), 2)

                # Mid spine (between hips and shoulders)
                spine_top = [(l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2]
                spine_bottom = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2]
                cv2.line(frame, tuple(map(int, spine_top)), tuple(map(int, spine_bottom)), (50, 255, 50), 2)
            except:
                pass

        if save_output:
            out.write(frame)

    cv2.imshow("Pose Tracker", frame)

    key = cv2.waitKey(20) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused
    elif paused and key == 83:  # Right arrow (->)
        paused = False
    elif paused and key == 81:  # Left arrow (<-)
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) - 2)
        paused = False

cap.release()
if save_output:
    out.release()
cv2.destroyAllWindows()