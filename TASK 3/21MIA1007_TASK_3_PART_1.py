import cv2
import os

# Paths for the image, video, and Haar Cascade files
image_path = r"C:\Users\tjtha\Downloads\ahah\simbhu.jpeg"
video_path = 'suspect.mp4'
output_image_path = 'processed_face.jpg'
output_video_path = 'processed_suspect.mp4'
output_faces_folder = 'processed_frame'

# Create the output folder for individual face images if it doesn't exist
os.makedirs(output_faces_folder, exist_ok=True)

# Load Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier("C:\Users\tjtha\Downloads\haarcascades\haarcascades\haarcascade_frontalface_alt.xml")

# Function to detect faces and draw rectangles
def detect_faces(frame, frame_number):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    face_count = 1  # Counter for face images in the current frame

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract and save each detected face as a separate image
        face_img = frame[y:y + h, x:x + w]
        face_filename = os.path.join(output_faces_folder, f"face{face_count}_frame{frame_number}.jpg")
        cv2.imwrite(face_filename, face_img)
        face_count += 1

    return frame

# Process the reference image
image = cv2.imread(image_path)
if image is not None:
    processed_image = detect_faces(image, frame_number=0)  # frame_number is 0 for reference image
    cv2.imwrite(output_image_path, processed_image)
    cv2.imshow("Processed Reference Image", cv2.resize(processed_image, (600, 600)))

# Process the video frame by frame, reading one frame per second
video = cv2.VideoCapture(video_path)
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# Set up video writer with matching FPS and frame size
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_path, fourcc, 1, (frame_width, frame_height))  # Save output at 1 FPS

frame_count = 0  # Track frame processing
processed_frames = 0  # Track the number of frames written

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Process every nth frame based on fps
    if frame_count % fps == 0:  # Only process 1 frame per second
        # Detect faces in this frame
        processed_frame = detect_faces(frame, frame_number=processed_frames + 1)
        output_video.write(processed_frame)
        
        # Display the processed frame
        resized_frame = cv2.resize(processed_frame, (600, 600))
        cv2.imshow("Processed Video - Suspect", resized_frame)
        processed_frames += 1
        print(f"Processed frame: {processed_frames}")

    frame_count += 1  # Increment frame count

    # Press 'q' to exit video display
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release resources
video.release()
output_video.release()
cv2.destroyAllWindows()

print(f"Total frames processed at 1 frame per second: {processed_frames}")