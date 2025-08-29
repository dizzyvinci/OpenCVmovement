import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
from datetime import datetime

# ---- TIMESTAMPED OUTPUT DIRECTORY ---- #
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir = f"analysis_{timestamp}"
os.makedirs(base_dir, exist_ok=True)
os.makedirs(f"{base_dir}/graphs", exist_ok=True)

# ---- SETTINGS ---- #
video_path = "recordings/pose_output_srvclip.mp4"
output_csv = f"{base_dir}/srv_hand_joint_angles.csv"
graph_dir = f"{base_dir}/graphs"
ml_data_output = f"{base_dir}/srv_hand_joint_data_ml_ready.csv"
clustered_video_output = f"{base_dir}/srv_pose_with_clusters.mp4"
n_clusters = 3

# ---- EXTRACT JOINT ANGLES ---- #
print("📥 Extracting joint angle data from video...")
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose()
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(3))
height = int(cap.get(4))

frame_data = []
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb)
    hand_results = hands.process(rgb)
    time_sec = frame_idx / fps

    row = {'frame': frame_idx, 'time_sec': time_sec}

    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark

        def get_point(name):
            lm = landmarks[mp_pose.PoseLandmark[name]]
            return np.array([lm.x * width, lm.y * height])

        try:
            # Elbow angles
            for side in ["LEFT", "RIGHT"]:
                shoulder = get_point(f"{side}_SHOULDER")
                elbow = get_point(f"{side}_ELBOW")
                wrist = get_point(f"{side}_WRIST")
                index = get_point(f"{side}_INDEX")

                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                wrist_angle = calculate_angle(elbow, wrist, index)
                row[f"{side}_ELBOW"] = elbow_angle
                row[f"{side}_WRIST"] = wrist_angle

        except:
            pass

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            joint_names = ["MCP", "PIP", "DIP", "TIP"]
            base_index = {
                4: "THUMB", 8: "INDEX", 12: "MIDDLE",
                16: "RING", 20: "PINKY"
            }
            for tip_idx, finger in base_index.items():
                for offset, joint in zip([0, -3, -2, -1], joint_names):
                    idx = tip_idx + offset
                    lm = hand_landmarks.landmark[idx]
                    row[f"{finger}_{joint}_x"] = lm.x * width
                    row[f"{finger}_{joint}_y"] = lm.y * height

    frame_data.append(row)
    frame_idx += 1

cap.release()

# ---- SAVE JOINT ANGLE CSV ---- #
df = pd.DataFrame(frame_data)
df.to_csv(output_csv, index=False)
print(f"✅ Saved joint angle CSV: {output_csv}")

# ---- GRAPHING ---- #
print("📊 Plotting joint angles...")
df_clean = df.dropna()
for col in df.columns:
    if col not in ['frame', 'time_sec', 'Cluster'] and df[col].notna().any():
        plt.figure(figsize=(10, 4))
        plt.plot(df['time_sec'], df[col], label=col)
        plt.xlabel("Time (s)")
        plt.ylabel("Angle / Position")
        plt.title(f"{col} over Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{graph_dir}/{col}.png")
        plt.close()

# ---- ACTIVITY STATS ---- #
print("📈 Analyzing joint activity levels...")
variances = df_clean.var(numeric_only=True).sort_values(ascending=False)
most_active = variances.head(5)
print("Most active joints (by variance):")
print(most_active)

# ---- ML CLUSTERING ---- #
print("🤖 Running KMeans clustering...")
feature_cols = [col for col in df_clean.columns if col not in ['frame', 'time_sec']]
X = df_clean[feature_cols].values

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X)

df_clean['Cluster'] = clusters
df['Cluster'] = np.nan
df.loc[df_clean.index, 'Cluster'] = clusters
df.to_csv(ml_data_output, index=False)
print(f"✅ ML data saved: {ml_data_output}")

# ---- CLUSTER-LABELED VIDEO ---- #
print("🎥 Generating cluster-labeled video...")
cap = cv2.VideoCapture(video_path)
out = cv2.VideoWriter(clustered_video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx < len(df):
        time_sec = df.loc[frame_idx, 'time_sec']
        cluster = df.loc[frame_idx, 'Cluster']
        label = f"Time: {time_sec:.2f}s"
        if not pd.isna(cluster):
            label += f" | Cluster: {int(cluster)}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print(f"✅ Cluster-labeled video saved: {clustered_video_output}")

# ---- ANGLE HELPER ---- #
def calculate_angle(a, b, c):
    a, b, c = map(np.array, [a, b, c])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle
