import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Helper function to calculate angles between 3 points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle

# Load your video here — update the path as needed
cap = cv2.VideoCapture("recordings/DRAAGON.mp4")  # Or "recordings/yourfile.mp4"

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run pose detection
    results = pose.process(rgb)

    if results.pose_landmarks:
        # Draw pose skeleton
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS)

        # Label all 33 joints
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            h, w = frame.shape[:2]
            x, y = int(lm.x * w), int(lm.y * h)
            joint_name = mp_pose.PoseLandmark(idx).name
            cv2.putText(frame, joint_name, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

        # Get key joints for left elbow angle
        landmarks = results.pose_landmarks.landmark
        h, w = frame.shape[:2]

        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * w,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * h]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * w,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * h]

        angle = calculate_angle(shoulder, elbow, wrist)

        # Draw angle at elbow
        cv2.putText(frame, f'{int(angle)}°', (int(elbow[0]), int(elbow[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Show result
    cv2.imshow('Pose Tracker', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
