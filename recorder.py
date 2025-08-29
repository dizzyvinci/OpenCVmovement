import cv2
import os
from datetime import datetime

# Create output folder
output_dir = "recordings"
os.makedirs(output_dir, exist_ok=True)

# Use current timestamp as filename
filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
filepath = os.path.join(output_dir, filename)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam")
    exit()

# Get width/height of the webcam feed
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define codec and create VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(filepath, fourcc, 20.0, (frame_width, frame_height))

print(f"Recording to: {filepath}")
print("Press Q to stop recording.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)
    cv2.imshow('Recording... Press Q to stop', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
