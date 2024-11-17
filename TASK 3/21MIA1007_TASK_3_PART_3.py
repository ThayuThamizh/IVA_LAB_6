import cv2
import numpy as np
import pandas as pd
import os

# Paths for the video and Haar Cascade files
video_path = 'processed_suspect.mp4'
output_video_path = 'processed_output_suspect.mp4'
output_faces_folder = 'processed_frame'

# Load Haar Cascade classifiers for facial feature detection
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_nose.xml')

# Load reference facial features from Excel
reference_features = pd.read_excel('facial_features.xlsx').iloc[0].to_dict()

# Directory for processed frames
selected_frames_folder = 'selected_frames'
if not os.path.exists(selected_frames_folder):
    os.makedirs(selected_frames_folder)

# Function to extract facial features
def extract_facial_features(image_path):
    features = {}
    frame = cv2.imread(image_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_region = gray[y:y + h, x:x + w]

        # Edge Detection (Sobel)
        edges = cv2.Sobel(face_region, cv2.CV_64F, 1, 0, ksize=5)
        edge_count = np.sum(edges > 0)  # Total edge pixels
        features['Edge_Count'] = edge_count

        # Geometric Features - Face Dimensions
        features['Face_Width'] = w
        features['Face_Height'] = h

        # Detect Eyes within the face region
        eyes = eye_cascade.detectMultiScale(face_region, 1.1, 4)
        if len(eyes) >= 2:
            eye_1, eye_2 = eyes[0], eyes[1]
            eye_spacing = abs((eye_1[0] + eye_1[2] / 2) - (eye_2[0] + eye_2[2] / 2))
            eye_width_avg = (eye_1[2] + eye_2[2]) / 2
            features['Eye_Spacing'] = eye_spacing
            features['Eye_Width_Avg'] = eye_width_avg
        else:
            features['Eye_Spacing'] = None
            features['Eye_Width_Avg'] = None

        # Detect Nose within the face region
        noses = nose_cascade.detectMultiScale(face_region, 1.1, 4)
        if len(noses) > 0:
            nose = noses[0]
            features['Nose_Width'] = nose[2]
            features['Nose_Height'] = nose[3]
        else:
            features['Nose_Width'] = None
            features['Nose_Height'] = None

    return features

# Function to calculate similarity score between extracted features and reference features
def calculate_similarity_score(extracted_features, ref_features):
    score = 0
    total_features = 0

    for key in ref_features:
        if key in extracted_features and extracted_features[key] is not None and ref_features[key] is not None:
            diff = abs(extracted_features[key] - ref_features[key])
            if ref_features[key] != 0:
                score += diff / ref_features[key]
            total_features += 1

    similarity_score = 1 - (score / total_features) if total_features > 0 else 0
    return similarity_score

# Process the video and calculate similarity score for each face in each frame
video = cv2.VideoCapture(video_path)
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# Set up video writer with matching FPS and frame size
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_path, fourcc, 1, (frame_width, frame_height))  # 1 FPS for output

frame_count = 0  # Track frame processing
processed_frames = 0  # Track the number of frames written

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Process every nth frame based on fps (1 frame per second)
    if frame_count % fps == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for face_number, (x, y, w, h) in enumerate(faces, start=1):
            face_region = frame[y:y + h, x:x + w]
            face_filename = os.path.join(output_faces_folder, f"face{face_number}_frame{processed_frames + 1}.jpg")
            cv2.imwrite(face_filename, face_region)
            
            # Extract features for the current face image
            extracted_features = extract_facial_features(face_filename)
            
            # Calculate similarity score with reference features
            similarity_score = calculate_similarity_score(extracted_features, reference_features)
            print(f"Frame {processed_frames + 1} - Face {face_number}: Similarity Score: {similarity_score:.2f}")
            
            # Draw bounding box and label based on similarity score
            if similarity_score >= 0.3:
                # Red box for suspect
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)  # Red
                cv2.putText(frame, "Suspect", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                # Green box for non-suspect
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Green
                cv2.putText(frame, "Non-Suspect", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        output_video.write(frame)
        processed_frames += 1

    frame_count += 1  # Increment frame count

    # Press 'q' to exit video display
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release resources
video.release()
output_video.release()
cv2.destroyAllWindows()

print(f"Total frames processed at 1 frame per second: {processed_frames}")