#TASK 1
import cv2
import numpy as np

# Load video
video_path = '/Users/thayu_task_1/Downloads/IVA_task_1_clip.mp4'
cap = cv2.VideoCapture(video_path)

# Background subtractor for initial detection
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

# Parameters for ShiTomasi corner detection (for selecting points to track)
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize variables for tracking
person_position = None
person_histogram = None
tracking_points = None
previous_frame_gray = None

# Function to calculate color histogram for appearance matching
def get_color_histogram(frame, bbox):
    x, y, w, h = bbox
    person_region = frame[y:y+h, x:x+w]
    hist = cv2.calcHist([person_region], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist

# Function to initialize tracking points within the bounding box
def initialize_tracking_points(frame_gray, bbox):
    x, y, w, h = bbox
    mask = np.zeros_like(frame_gray)
    mask[y:y+h, x:x+w] = 255  # Define the region to search for features
    points = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
    return points

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for optical flow tracking
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Initial detection using background subtraction
    if person_position is None:
        # Apply background subtraction
        mask = bg_subtractor.apply(frame)
        
        # Perform morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_area = 0
        best_bbox = None

        for cnt in contours:
            # Ignore small contours
            area = cv2.contourArea(cnt)
            if area < 1000:
                continue
            
            # Get bounding box of the largest contour, assuming it's the person
            x, y, w, h = cv2.boundingRect(cnt)
            if area > max_area:
                max_area = area
                best_bbox = (x, y, w, h)
        
        if best_bbox:
            # Set the initial bounding box and tracking points
            person_histogram = get_color_histogram(frame, best_bbox)
            person_position = best_bbox
            tracking_points = initialize_tracking_points(frame_gray, best_bbox)
            previous_frame_gray = frame_gray.copy()
    
    else:
        # Track the points with optical flow
        if tracking_points is not None and previous_frame_gray is not None:
            # Calculate optical flow
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(previous_frame_gray, frame_gray, tracking_points, None, **lk_params)
            
            # Select good points
            good_new = next_points[status == 1]
            good_old = tracking_points[status == 1]
            
            # Update bounding box based on movement of points
            x_movement = np.mean(good_new[:, 0] - good_old[:, 0])
            y_movement = np.mean(good_new[:, 1] - good_old[:, 1])
            
            x, y, w, h = person_position
            x = int(x + x_movement)
            y = int(y + y_movement)
            person_position = (x, y, w, h)
            
            # Draw the updated bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Update points and previous frame for the next iteration
            tracking_points = good_new.reshape(-1, 1, 2)
            previous_frame_gray = frame_gray.copy()
        else:
            # If tracking is lost, reset tracking to find person again
            person_position = None

    # Display the frame
    cv2.imshow("Person Tracking with Optical Flow", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()