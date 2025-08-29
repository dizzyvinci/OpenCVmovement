import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---- SETTINGS ---- #
video_path = "recordings/pose_output_srvclip.mp4"
output_csv = "analysis/srv_hand_joint_angles.csv"
graph_dir = "analysis/graphs"
ml_data_output = "analysis/srv_hand_joint_data_ml_ready.csv"

os.makedirs("analysis", exist_ok=True)
os.makedirs(graph_dir, exist_ok=True)

# ---- INIT ---- #
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
frame_idx = 0
data = []

# ---- HELPERS ---- #
def calculate_angle(a, b, c):
    a, b, c = map(np.array, [a, b, c])
    ab = b - a
    cb = b - c
    radians = np.arccos(np.clip(np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb)), -1.0, 1.0))
    return np.degrees(radians)

def landmark_to_coords(landmarks, h, w):
    return [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

# ---- MAIN LOOP ---- #
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    h, w = frame.shape[:2]
    row = {'frame': frame_idx, 'time_sec': frame_idx / fps}

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        coords = landmark_to_coords(hand_landmarks.landmark, h, w)

        # Each finger: [MCP, PIP, DIP, TIP]
        fingers = {
            "Thumb": [1, 2, 3, 4],
            "Index": [5, 6, 7, 8],
            "Middle": [9, 10, 11, 12],
            "Ring": [13, 14, 15, 16],
            "Pinky": [17, 18, 19, 20]
        }

        for name, ids in fingers.items():
            try:
                mcp_angle = calculate_angle(coords[ids[0]-1], coords[ids[0]], coords[ids[1]])
                pip_angle = calculate_angle(coords[ids[0]], coords[ids[1]], coords[ids[2]])
                dip_angle = calculate_angle(coords[ids[1]], coords[ids[2]], coords[ids[3]])
                row[f'{name}_MCP'] = mcp_angle
                row[f'{name}_PIP'] = pip_angle
                row[f'{name}_DIP'] = dip_angle
            except:
                row[f'{name}_MCP'] = None
                row[f'{name}_PIP'] = None
                row[f'{name}_DIP'] = None

    data.append(row)
    frame_idx += 1

cap.release()
hands.close()

# ---- SAVE TO CSV ---- #
df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)
print(f"✅ Saved joint angle data to: {output_csv}")

# ---- GENERATE GRAPHS ---- #
for col in df.columns:
    if col not in ['frame', 'time_sec'] and df[col].notna().any():
        plt.figure(figsize=(10, 4))
        plt.plot(df['time_sec'], df[col], label=col)
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (°)")
        plt.title(f"{col} over time")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{graph_dir}/{col}.png")
        plt.close()

print(f"✅ Saved joint angle graphs to: {graph_dir}")

# ---- EXPORT FOR ML ---- #
ml_df = df.drop(columns=["frame"])
ml_df = ml_df.dropna()  # drop rows with missing joint info
ml_df.to_csv(ml_data_output, index=False)
print(f"✅ Saved ML-ready data to: {ml_data_output}")
