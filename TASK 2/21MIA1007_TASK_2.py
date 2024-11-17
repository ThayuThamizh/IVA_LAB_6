import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Set up directories for each step's output
os.makedirs('frames', exist_ok=True)
os.makedirs('differenced_frames', exist_ok=True)
os.makedirs('threshold_frames', exist_ok=True)
os.makedirs('detected_people', exist_ok=True)
os.makedirs('peak_frames', exist_ok=True)

# Load videoq
video = cv2.VideoCapture("task2_video.mp4")
fps = video.get(cv2.CAP_PROP_FPS)  # Frames per second (assumed 30 FPS)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
interval = int(5 * fps)  # Interval of 5 seconds, i.e., 150 frames at 30 FPS

# Initialize Background Subtractor for stationary object detection
back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

# Step 1: Save Original Frames
frame_count = 0
people_count_per_frame = []
_, prev_frame = video.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    
    # Save the original frame
    cv2.imwrite(f'frames/frame_{frame_count}.jpg', frame)
    
    # Step 2: Differencing for Motion Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff_frame = cv2.absdiff(prev_gray, gray)
    cv2.imwrite(f'differenced_frames/diff_frame_{frame_count}.jpg', diff_frame)

    # Step 3: Apply Thresholding to Highlight Motion
    _, thresh = cv2.threshold(diff_frame, 25, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f'threshold_frames/thresh_frame_{frame_count}.jpg', thresh)

    # Step 4: Background Subtraction for Stationary People Detection
    fg_mask = back_sub.apply(frame)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    dilated_fg = cv2.dilate(fg_mask, None, iterations=2)

    # Step 5: Contour Detection to Count People (Both Moving and Stationary)
    combined_mask = cv2.bitwise_or(thresh, dilated_fg)  # Combine motion and stationary masks
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    for c in contours:
        if cv2.contourArea(c) > 500:  # Filter small areas
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            count += 1

    people_count_per_frame.append(count)
    cv2.imwrite(f'detected_people/people_frame_{frame_count}.jpg', frame)
    
    prev_gray = gray
    frame_count += 1

# Step 6: Identify Peak Interval
people_count_per_interval = [
    sum(people_count_per_frame[i:i + interval]) for i in range(0, len(people_count_per_frame), interval)
]

# Plot the results
plt.plot(people_count_per_interval)
plt.xlabel('5-Second Intervals')
plt.ylabel('Total People Count')
plt.title('People Count Over Time')
plt.show()

# Find peak interval
peak_interval = np.argmax(people_count_per_interval)
print(f"Peak shopping duration is in interval {peak_interval}, with the highest count of people.")

# Step 7: Save Frames from Peak Interval for Visual Confirmation
peak_start_frame = peak_interval * interval
video.set(cv2.CAP_PROP_POS_FRAMES, peak_start_frame)

frame_count = 0
for _ in range(int(fps * 5)):  # Save frames for 5 seconds
    ret, frame = video.read()
    if not ret:
        break
    cv2.imwrite(f'peak_frames/peak_frame_{frame_count}.jpg', frame)
    frame_count += 1

# Cleanup
video.release()
cv2.destroyAllWindows()