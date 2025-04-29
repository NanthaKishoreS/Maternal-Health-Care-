import cv2
import numpy as np

# Load image
img = cv2.imread('eye_photo.png')

# Optional: Crop manually or use face/eye detection (simple crop for now)
cropped_eye = img[100:300, 200:400]  # example coordinates (you can adjust)

# Convert to LAB color space (better for color analysis)
lab = cv2.cvtColor(cropped_eye, cv2.COLOR_BGR2LAB)

# 'b' channel in LAB represents yellow/blue
l, a, b = cv2.split(lab)

# Calculate average 'b' value
avg_b = np.mean(b)

# Define threshold for jaundice detection (this is experimental)
threshold_b = 135

if avg_b > threshold_b:
    print("Possible jaundice detected! (b-value:", avg_b, ")")
else:
    print("No jaundice detected. (b-value:", avg_b, ")")
