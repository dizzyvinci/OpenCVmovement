import cv2
import mediapipe as mp
import numpy as np

# ---- SETTINGS ---- #
input_path = "vidDownloads/Stevie Ray Vaughan - Texas Flood (Live at the El Mocambo).mp4"
output_path = "recordings/pose_output_srvclip.mp4"

mirror_input = True
save_output = True
show_hand_names = True

# ---- OPTIONAL CLIP TRIM ---- #
enable_clip_trimming = True
clip_start_sec = 33
clip_end_sec = 40

# ---- INIT ---- #
cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(3))
height = int(cap.get(4))

if enable_clip_trimming:
    cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start_sec * fps)

if save_output:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

def calculate_angle(a, b, c):
    a, b, c = map(np.array, [a, b, c])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

paused = False

# ---- MAIN LOOP ---- #
while cap.isOpened():
    current_time_sec = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
    if enable_clip_trimming and current_time_sec >= clip_end_sec:
        break

    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        if mirror_input:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        pose_results = pose.process(rgb)
        hand_results = hands.process(rgb)

        # ---- POSE LANDMARKS ---- #
        if pose_results.pose_landmarks:
            mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = pose_results.pose_landmarks.landmark

            def get_point(name):
                lm = landmarks[mp_pose.PoseLandmark[name]]
                return [lm.x * w, lm.y * h]

            try:
                # Extract landmarks
                l_shoulder = get_point("LEFT_SHOULDER")
                l_elbow = get_point("LEFT_ELBOW")
                l_wrist = get_point("LEFT_WRIST")
                r_shoulder = get_point("RIGHT_SHOULDER")
                r_elbow = get_point("RIGHT_ELBOW")
                r_wrist = get_point("RIGHT_WRIST")

                l_index = get_point("LEFT_INDEX")
                r_index = get_point("RIGHT_INDEX")

                # Elbow angles
                l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
                cv2.putText(frame, f'{int(l_angle)}°', tuple(map(int, l_elbow)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(frame, f'{int(r_angle)}°', tuple(map(int, r_elbow)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                # Wrist angles
                l_wrist_angle = calculate_angle(l_elbow, l_wrist, l_index)
                r_wrist_angle = calculate_angle(r_elbow, r_wrist, r_index)
                cv2.putText(frame, f'{int(l_wrist_angle)}°', tuple(map(int, l_wrist)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                cv2.putText(frame, f'{int(r_wrist_angle)}°', tuple(map(int, r_wrist)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                # Neck angle
                nose = get_point("NOSE")
                neck_angle = calculate_angle(l_shoulder, nose, r_shoulder)
                center = (int((l_shoulder[0] + r_shoulder[0]) / 2), int((l_shoulder[1] + r_shoulder[1]) / 2))
                cv2.putText(frame, f'{int(neck_angle)}°', (center[0], center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 100, 255), 2)

                # Spine overlay
                l_hip = get_point("LEFT_HIP")
                r_hip = get_point("RIGHT_HIP")
                spine_top = [(l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2]
                spine_bottom = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2]

                cv2.line(frame, tuple(map(int, l_shoulder)), tuple(map(int, l_hip)), (100, 200, 255), 2)
                cv2.line(frame, tuple(map(int, r_shoulder)), tuple(map(int, r_hip)), (100, 200, 255), 2)
                cv2.line(frame, tuple(map(int, l_hip)), tuple(map(int, r_hip)), (0, 200, 200), 2)
                cv2.line(frame, tuple(map(int, spine_top)), tuple(map(int, spine_bottom)), (50, 255, 50), 2)

            except:
                pass

        # ---- HAND TRACKING ---- #
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                if show_hand_names:
                    for idx, lm in enumerate(hand_landmarks.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.putText(frame, f"{idx}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

        # ---- TIMESTAMP ---- #
        timestamp = f"⏱ {current_time_sec:.2f}s"
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if save_output:
            out.write(frame)

    cv2.imshow("Pose Tracker - Advanced", frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused
    elif key == ord('d') or key == 83:
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + (fps * 5))
    elif key == ord('a') or key == 81:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cap.get(cv2.CAP_PROP_POS_FRAMES) - (fps * 5)))
    elif key == ord('.') and paused:
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 1)
    elif key == ord(',') and paused:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cap.get(cv2.CAP_PROP_POS_FRAMES) - 2))

cap.release()
if save_output:
    out.release()
cv2.destroyAllWindows()
