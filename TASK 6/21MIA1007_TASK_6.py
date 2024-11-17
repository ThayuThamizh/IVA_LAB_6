import cv2
import numpy as np
import os

# Path for video
video_path = 'cars24.mp4'
# Path for Haar cascade
cascade_path = 'haarcascade_cars.xml'
# Output folder for frames with black cars
output_folder = 'black_car_frames'
os.makedirs(output_folder, exist_ok=True)

# Initialize video capture and load the car cascade
cap = cv2.VideoCapture(video_path)
car_cascade = cv2.CascadeClassifier(cascade_path)

black_car_count = 0  # Counter for black cars
frame_count = 0      # Frame counter

# Define HSV range for black color
lower_black_hsv = np.array([0, 0, 0])
upper_black_hsv = np.array([180, 255, 60])  # Adjusted HSV range

while True:
    # Capture frame by frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale (required by the cascade classifier)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars in the frame
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    # Variable to check if a black car is detected in this frame
    black_car_detected = False

    # Draw rectangle around each detected car and check for black color
    for (x, y, w, h) in cars:
        # Draw a green bounding box around each detected car
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Extract the car region of interest (ROI) from the frame
        car_roi = frame[y:y+h, x:x+w]

        # Convert the car ROI to HSV color space for color detection
        hsv_roi = cv2.cvtColor(car_roi, cv2.COLOR_BGR2HSV)

        # Create a mask to detect black color in the car ROI
        black_mask = cv2.inRange(hsv_roi, lower_black_hsv, upper_black_hsv)
        black_pixels = cv2.countNonZero(black_mask)
        total_pixels = w * h
        black_ratio = black_pixels / total_pixels

        # Display black_ratio for debugging purposes
        print(f"Car at ({x}, {y}), Black Ratio: {black_ratio:.2f}")

        # If black ratio is above the threshold, it's likely a black car
        if black_ratio > 0.3:  # Adjust the threshold as needed
            # Draw a blue bounding box for black cars
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            black_car_count += 1
            black_car_detected = True
            print(f"Black car detected: {black_car_count}")

    # Save the frame if a black car is detected
    if black_car_detected:
        frame_output_path = os.path.join(output_folder, f'black_car_frame_{frame_count}.jpg')
        cv2.imwrite(frame_output_path, frame)

    # Display the frame with bounding boxes
    cv2.imshow('Detected Cars', frame)
    frame_count += 1

    # Press 'q' to exit the video
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the video capture object and close all frames
cap.release()
cv2.destroyAllWindows()

print(f"Total black cars detected: {black_car_count}")