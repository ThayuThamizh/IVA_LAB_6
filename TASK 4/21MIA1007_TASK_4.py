import cv2
import os

# Create output directory if not exists
output_dir = 'people_count_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize video capture
cap = cv2.VideoCapture('21MIA1007_TASK_4.mp4')  
# Initialize variables
ret, frame1 = cap.read()
ret, frame2 = cap.read()
entrance_count = 0
exit_count = 0

# Define the ROIs for entrance and exit
# Adjust these coordinates based on the revolving door's position in the video
entrance_roi_x, entrance_roi_y, entrance_roi_w, entrance_roi_h = 50, 200, 100, 200  # Example coordinates for entrance ROI
exit_roi_x, exit_roi_y, exit_roi_w, exit_roi_h = 300, 200, 100, 200  # Example coordinates for exit ROI

while ret:
    # Calculate the difference between two consecutive frames
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the ROIs on the frame
    cv2.rectangle(frame1, (entrance_roi_x, entrance_roi_y), 
                  (entrance_roi_x + entrance_roi_w, entrance_roi_y + entrance_roi_h), (0, 255, 0), 2)
    cv2.rectangle(frame1, (exit_roi_x, exit_roi_y), 
                  (exit_roi_x + exit_roi_w, exit_roi_y + exit_roi_h), (0, 0, 255), 2)

    for contour in contours:
        # Ignore small contours
        if cv2.contourArea(contour) < 1500:
            continue

        # Get bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Check if the contour is in the entrance ROI
        if entrance_roi_x < x < entrance_roi_x + entrance_roi_w and entrance_roi_y < y < entrance_roi_y + entrance_roi_h:
            entrance_count += 1
            cv2.putText(frame1, "Entered", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Check if the contour is in the exit ROI
        elif exit_roi_x < x < exit_roi_x + exit_roi_w and exit_roi_y < y < exit_roi_y + exit_roi_h:
            exit_count += 1
            cv2.putText(frame1, "Exited", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display counts
    cv2.putText(frame1, f"Entered: {entrance_count}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame1, f"Exited: {exit_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Save the current frame with annotations to the output directory
    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cv2.imwrite(f"{output_dir}/frame_{frame_number}.jpg", frame1)

    # Show the frame (optional)
    cv2.imshow("People Counting", frame1)

    # Prepare for the next frame
    frame1 = frame2
    ret, frame2 = cap.read()

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print(f"Total people entered: {entrance_count}")
print(f"Total people exited: {exit_count}")