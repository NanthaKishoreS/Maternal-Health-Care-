import cv2
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

# Open the webcam (camera ID 0 is usually the default)
cap = cv2.VideoCapture(0)

# Lists to store average red channel values and timestamps
red_values = []
times = []

# Sampling rate (frames per second)
fps = 30  

print("Place your finger gently over the camera lens...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip if needed
    frame = cv2.flip(frame, 1)

    # Get the red channel average
    red_channel = frame[:, :, 2]
    avg_red = np.mean(red_channel)
    red_values.append(avg_red)
    times.append(len(times) / fps)

    # Show preview window
    cv2.imshow('Camera Preview - Cover Lens with Finger', frame)

    # Exit if 'ESC' is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Convert lists to numpy arrays
red_values = np.array(red_values)
times = np.array(times)

# Bandpass filter to focus on human pulse range (~0.7 Hz to ~4 Hz)
b, a = scipy.signal.butter(3, [0.7/(0.5*fps), 4/(0.5*fps)], btype='band')
filtered = scipy.signal.filtfilt(b, a, red_values)

# Find dominant frequency using FFT
fft = np.fft.fft(filtered)
freqs = np.fft.fftfreq(len(filtered), 1/fps)

# Only consider positive frequencies
idx = np.where(freqs > 0)
freqs = freqs[idx]
fft = np.abs(fft[idx])

# Find the frequency with the maximum power
peak_freq = freqs[np.argmax(fft)]

# Convert Hz to Beats Per Minute (BPM)
bpm = peak_freq * 60

print(f"Estimated Pulse: {bpm:.2f} BPM")

# Optional: plot the filtered signal
plt.plot(times, filtered)
plt.title(f'Filtered Pulse Signal (Estimated BPM: {bpm:.2f})')
plt.xlabel('Time (s)')
plt.ylabel('Filtered Red Intensity')
plt.grid()
plt.show()
