import cv2
import numpy as np
import scipy.signal
import time

# === CONFIGURATION ===
video_path = '/home/nantha-kishore-s/rppg-project/sample_video.mp4'  # <-- set your correct video filename here
required_frames = 300  # About 10 seconds of video if 30 FPS

# === LOAD VIDEO ===
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

print("Processing video...")

red_values = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended or failed to grab frame.")
        break

    # Resize frame if too large (optional)
    frame = cv2.resize(frame, (640, 480))

    # Get average RED channel intensity
    red_channel = frame[:, :, 2]
    avg_red = np.mean(red_channel)
    red_values.append(avg_red)

    # Show the frame (optional)
    cv2.imshow('Frame', frame)

    # Stop manually if needed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Manually stopped.")
        break

    # For short videos: break if enough frames collected
    if len(red_values) >= required_frames:
        print(f"Collected {len(red_values)} frames, processing...")
        break

cap.release()
cv2.destroyAllWindows()

# === PROCESS SIGNAL ===
red_values = np.array(red_values)
print(f"Total frames collected: {len(red_values)}")

if len(red_values) < 30:
    print("Not enough data points. Try recording a longer video.")
    exit()

# === FILTERING ===
b, a = scipy.signal.butter(3, [0.5 / 30, 5 / 30], btype='bandpass')
try:
    filtered = scipy.signal.filtfilt(b, a, red_values)
except ValueError as e:
    print(f"Filtering error: {e}")
    exit()

# === PEAK DETECTION ===
peaks, _ = scipy.signal.find_peaks(filtered, distance=15)

# === HEART RATE CALCULATION ===
duration_sec = len(red_values) / 30  # Assuming 30 FPS
num_beats = len(peaks)
heart_rate = (num_beats / duration_sec) * 60

print(f"Estimated Heart Rate: {heart_rate:.2f} BPM")
