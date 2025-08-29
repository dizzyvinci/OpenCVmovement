import cv2
import mediapipe as mp
import numpy as np

# ---- SETTINGS ---- #
input_path = "recordings/DRAAGON.mp4"  # Or 0 for webcam
output_path = "recordings/pose_output.mp4"

mirror_input = True          # Set to False if no mirroring needed
save_output = True           # Set to False to disable video recording
show_hand_names = True       # Show both index + name on hand joints

# ---- INIT ---- #
cap = cv2.VideoCapture(input_path)

if save_output:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0,
                          (int(cap.get(3)), int(cap.get(4))))

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose()
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

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
        hand_results = hands.process(rgb)
        h, w = frame.shape[:2]

        # ---- POSE ---- #
        if pose_results.pose_landmarks:
            mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            for idx, lm in enumerate(pose_results.pose_landmarks.landmark):
                x, y = int(lm.x * w), int(lm.y * h)
                joint_name = mp_pose.PoseLandmark(idx).name
                cv2.putText(frame, joint_name, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

            # Elbow angle (LEFT)
            lms = pose_results.pose_landmarks.landmark
            shoulder = [lms[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w,
                        lms[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h]
            elbow = [lms[mp_pose.PoseLandmark.LEFT_ELBOW].x * w,
                     lms[mp_pose.PoseLandmark.LEFT_ELBOW].y * h]
            wrist = [lms[mp_pose.PoseLandmark.LEFT_WRIST].x * w,
                     lms[mp_pose.PoseLandmark.LEFT_WRIST].y * h]

            angle = calculate_angle(shoulder, elbow, wrist)
            cv2.putText(frame, f'{int(angle)}°', (int(elbow[0]), int(elbow[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # ---- HANDS ---- #
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                for i, lm in enumerate(hand_landmarks.landmark):
                    x, y = int(lm.x * w), int(lm.y * h)
                    if show_hand_names:
                        joint_name = mp_hands.HandLandmark(i).name
                        label = f"{i} {joint_name}"
                    else:
                        label = str(i)
                    cv2.putText(frame, label, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)

        # ---- Output Video ---- #
        if save_output:
            out.write(frame)

    # ---- Show Frame ---- #
    cv2.imshow("Pose + Hand Tracker", frame)

    key = cv2.waitKey(20) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused
    elif paused and key == 83:  # Right arrow (→)
        paused = False
    elif paused and key == 81:  # Left arrow (←)
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) - 2)
        paused = False

# ---- Cleanup ---- #
cap.release()
if save_output:
    out.release()
cv2.destroyAllWindows()
