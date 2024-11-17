import cv2
import os
import time
import pandas as pd
import random

# Create folder to save results
output_folder = "frame_time_results"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load the video
video_path = 'task5.mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Initialize background subtractor
background_subtractor = cv2.createBackgroundSubtractorMOG2()

# Tracking data structures
presence_times = {}       # Stores the entry time of each detected person
object_times = {}         # Tracks the time each person spends in the frame
object_positions = {}     # Tracks the position of each object to prevent ID reassignment
object_colors = {}        # Stores a unique color for each detected person
object_id_counter = 1

# Function to check if a person is already being tracked
def match_existing_object(x, y, w, h):
    for obj_id, (ox, oy, ow, oh) in object_positions.items():
        if abs(x - ox) < 50 and abs(y - oy) < 50:  # Adjust threshold as needed
            object_positions[obj_id] = (x, y, w, h)  # Update position
            return obj_id
    return None

# Generate a random color
def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction to detect objects
    mask = background_subtractor.apply(frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    # Find contours of detected objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over contours to detect and track objects
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue

        # Get bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Check if this is an existing person or a new one
        obj_id = match_existing_object(x, y, w, h)
        if obj_id is None:
            # New person detected
            obj_id = object_id_counter
            presence_times[obj_id] = time.time()  # Log entry time
            object_positions[obj_id] = (x, y, w, h)  # Save position
            object_colors[obj_id] = generate_random_color()  # Assign a unique color
            object_id_counter += 1

        # Calculate the time spent in the frame
        time_in_frame = time.time() - presence_times[obj_id]
        object_times[obj_id] = time_in_frame

        # Display bounding box, ID, and time in frame with unique color
        color = object_colors[obj_id]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"Time: {time_in_frame:.1f}s", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, f"ID: {obj_id}", (x + w, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the frame with bounding boxes and time in frame
    cv2.imshow('Presence Time Detection', frame)

    # Break if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Save all presence times to a CSV file
presence_time_data = [{'Object ID': obj_id, 'Time in Frame (seconds)': time_in_frame} for obj_id, time_in_frame in object_times.items()]
df = pd.DataFrame(presence_time_data)
df.to_csv(os.path.join(output_folder, 'presence_times.csv'), index=False)
print("Presence times for all objects saved in presence_times.csv.")

# Release resources
cap.release()
cv2.destroyAllWindows()