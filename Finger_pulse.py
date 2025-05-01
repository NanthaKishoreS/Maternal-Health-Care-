import cv2
import numpy as np
import scipy.signal
import time
from collections import deque
import matplotlib.pyplot as plt

# Configuration
MIN_DURATION = 10    # Minimum measurement in seconds
MAX_DURATION = 35    # Maximum measurement time in seconds
MIN_BEATS = 12       # Minimum beats needed for reliable reading
BUFFER_SIZE = 7      # Smoothing buffer size (increased for stability)
WARM_UP_TIME = 3     # Initial warm-up period to discard (seconds)
EXPECTED_FPS = 30    # Expected camera FPS
HR_MIN = 40          # Minimum physiologically plausible heart rate
HR_MAX = 200         # Maximum physiologically plausible heart rate

def get_bpm():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return None
    
    # Attempt to set a higher resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Try to set higher FPS if supported
    cap.set(cv2.CAP_PROP_FPS, EXPECTED_FPS)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera running at {actual_fps:.1f} FPS")
    
    # Data collection
    timestamps, green_values = [], []
    hr_buffer = deque(maxlen=BUFFER_SIZE)
    start_time = time.time()
    last_hr_time = 0
    
    # Signal quality metrics
    signal_quality = "Initializing..."
    
    print("Place thumb on camera with light behind. Stay still...")
    print("Press 'q' to quit, 'r' to reset measurement")
    
    # For visualization
    recent_values = deque(maxlen=100)
    window_name = "Heart Rate Monitor"
    cv2.namedWindow(window_name)
    
    # Main capture loop
    while True:
        elapsed = time.time() - start_time
        
        # Check if maximum time has elapsed
        if elapsed > MAX_DURATION:
            print(f"Maximum measurement time ({MAX_DURATION}s) reached")
            break
        
        # Capture frame
        ret, frame = cap.read()
        if not ret: 
            print("Error: Couldn't read from camera")
            break
        
        # Create a copy for display
        display_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Define ROI (center of frame)
        roi_size = min(h, w) // 3
        x_center, y_center = w // 2, h // 2
        roi = frame[y_center-roi_size:y_center+roi_size, 
                   x_center-roi_size:x_center+roi_size]
        
        # Draw ROI rectangle on display frame
        cv2.rectangle(display_frame, 
                     (x_center-roi_size, y_center-roi_size),
                     (x_center+roi_size, y_center+roi_size),
                     (0, 255, 0), 2)
        
        # Extract color channels from ROI
        if roi.size > 0:  # Make sure ROI is not empty
            # Get average of each channel with bright pixel filtering
            b, g, r = cv2.split(roi)
            
            # Focus on green channel as it has best signal for blood volume changes
            green = np.mean(g)
            recent_values.append(green)
            
            # Only add data after warm-up period to avoid initial instability
            if elapsed > WARM_UP_TIME:
                green_values.append(green)
                timestamps.append(elapsed)
        
        # Process signal to calculate heart rate
        current_hr = None
        peaks = []
        filtered = []
        
        if len(green_values) > EXPECTED_FPS * 3:  # At least 3 seconds of data
            # Calculate actual sampling rate
            fps = len(green_values) / (timestamps[-1] - timestamps[0])
            
            # Normalize the signal
            normalized = np.array(green_values)
            normalized = (normalized - np.mean(normalized)) / np.std(normalized)
            
            # Design bandpass filter (0.7-3.5 Hz corresponds to 42-210 BPM)
            nyquist = fps / 2
            low, high = 0.7 / nyquist, 3.5 / nyquist
            if low < 1.0 and high < 1.0:  # Make sure frequencies are valid
                b, a = scipy.signal.butter(3, [low, high], btype='bandpass')
                filtered = scipy.signal.filtfilt(b, a, normalized)
                
                # Find peaks with dynamic threshold based on signal quality
                threshold = max(0.35, np.std(filtered) * 0.5)
                min_distance = int(fps / (HR_MAX / 60))  # Minimum samples between peaks
                
                peaks, properties = scipy.signal.find_peaks(
                    filtered,
                    distance=min_distance,
                    prominence=threshold,
                    height=threshold * 0.5
                )
                
                # Calculate heart rate if we have enough peaks
                if len(peaks) >= 2:
                    # Calculate intervals between peaks
                    intervals = np.diff(np.array(timestamps)[peaks])
                    
                    # Filter out physiologically implausible intervals
                    valid_intervals = intervals[(60/intervals >= HR_MIN) & (60/intervals <= HR_MAX)]
                    
                    if len(valid_intervals) >= 2:
                        # Calculate BPM from intervals
                        hr = 60 / np.median(valid_intervals)
                        
                        # Assess signal quality based on interval consistency
                        rmssd = np.sqrt(np.mean(np.square(np.diff(valid_intervals))))
                        quality_metric = rmssd / np.mean(valid_intervals)
                        
                        if quality_metric < 0.2:
                            signal_quality = "Good"
                        elif quality_metric < 0.4:
                            signal_quality = "Fair"
                        else:
                            signal_quality = "Poor - Keep still"
                        
                        # Only accept physiologically plausible readings
                        if HR_MIN <= hr <= HR_MAX:
                            current_hr = hr
                            hr_buffer.append(hr)
                            last_hr_time = elapsed
        
        # Visualization of the green channel over time
        if len(recent_values) > 2:
            # Create visualization area
            viz_height = 150
            viz = np.ones((viz_height, w, 3), dtype=np.uint8) * 255
            
            # Scale the values to fit in our visualization
            values_array = np.array(list(recent_values))
            min_val, max_val = np.min(values_array), np.max(values_array)
            scaled = (values_array - min_val) / (max_val - min_val + 1e-10) * (viz_height - 20)
            
            # Draw the signal
            points = [(i * w // len(recent_values), viz_height - int(val)) 
                     for i, val in enumerate(scaled)]
            for i in range(len(points) - 1):
                cv2.line(viz, points[i], points[i+1], (0, 200, 0), 2)
            
            # Mark detected peaks if available
            if len(filtered) > 0 and len(peaks) > 0:
                recent_idx = len(filtered) - len(recent_values)
                visible_peaks = [p for p in peaks if p >= recent_idx]
                for p in visible_peaks:
                    rel_idx = p - recent_idx
                    if 0 <= rel_idx < len(recent_values):
                        peak_x = rel_idx * w // len(recent_values)
                        cv2.circle(viz, (peak_x, viz_height - int(scaled[rel_idx])), 
                                  5, (0, 0, 255), -1)
            
            # Combine with the main display
            display_frame = np.vstack([display_frame, viz])
        
        # Display information on the frame
        y_pos = 30
        if len(hr_buffer) > 0 and elapsed - last_hr_time < 5:
            # Calculate median heart rate from buffer
            median_hr = np.median(hr_buffer)
            cv2.putText(display_frame, f"Heart Rate: {int(round(median_hr))} BPM", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Check if we have a stable enough reading
            if (len(hr_buffer) == BUFFER_SIZE and 
                np.std(list(hr_buffer)) < 5 and 
                elapsed > MIN_DURATION and
                signal_quality == "Good"):
                
                # Return the result if all criteria are met
                cap.release()
                cv2.destroyAllWindows()
                return int(round(median_hr))
        else:
            cv2.putText(display_frame, "Measuring...", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 2)
        
        # Show time remaining
        remaining = MAX_DURATION - elapsed
        if remaining < 10:  # Highlight time when less than 10 seconds left
            time_color = (0, 0, 255)  # Red when time is running out
        else:
            time_color = (0, 0, 0)
        
        # Show elapsed time and remaining time
        y_pos += 30
        cv2.putText(display_frame, f"Time: {elapsed:.1f}s (Remaining: {remaining:.1f}s)", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, time_color, 2)
        
        # Show signal quality
        y_pos += 30
        color = (0, 255, 0) if signal_quality == "Good" else \
                (0, 140, 255) if signal_quality == "Fair" else (0, 0, 255)
        cv2.putText(display_frame, f"Signal: {signal_quality}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Display the resulting frame
        cv2.imshow(window_name, display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset measurements
            timestamps, green_values = [], []
            hr_buffer.clear()
            start_time = time.time()
            print("Measurement reset")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    # If we have enough measurements, return the median
    if len(hr_buffer) > MIN_BEATS // 2:
        return int(round(np.median(hr_buffer)))
    
    # If we reached max time with some measurements but not enough for ideal confidence
    elif len(hr_buffer) > 0:
        print(f"Time limit reached. Returning best estimate from {len(hr_buffer)} measurements.")
        return int(round(np.median(hr_buffer)))
    
    return None

def analyze_heart_rate(bpm):
    """Provide a simple analysis of the heart rate"""
    if bpm is None:
        return "No valid heart rate measurement obtained"
    if bpm < 60:
        return f"BPM: {bpm} - Bradycardia (slow heart rate)"
    elif bpm <= 100:
        return f"BPM: {bpm} - Normal heart rate"
    else:
        return f"BPM: {bpm} - Tachycardia (elevated heart rate)"

if __name__ == "__main__":
    print("Heart Rate Monitor")
    print("Place your thumb on the camera with a light source behind it")
    print("Try to keep your hand steady during measurement")
    print(f"Maximum measurement time: {MAX_DURATION} seconds")
    
    bpm = get_bpm()
    
    if bpm is not None:
        result = analyze_heart_rate(bpm)
        print("\nMeasurement Complete!")
        print(result)
        
        # Display confidence level based on measurement quality
        if hasattr(get_bpm, 'hr_buffer'): 
            hr_buffer = get_bpm.hr_buffer
            if len(hr_buffer) == BUFFER_SIZE and np.std(list(hr_buffer)) < 5:
                print("Confidence: High")
            elif len(hr_buffer) >= MIN_BEATS // 2:
                print("Confidence: Medium")
            else:
                print("Confidence: Low - Consider retrying for better accuracy")
    else:
        print("\nMeasurement failed. Please try again with:")
        print("- Better lighting behind your thumb")
        print("- Keep your hand more steady")
        print("- Ensure your thumb properly covers the camera")