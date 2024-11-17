import cv2
import pandas as pd
import numpy as np

# Paths for the image, Haar Cascade files, and output Excel file
image_path = 'processed_face.jpg'
output_excel_path = 'facial_features.xlsx'

# Load Haar Cascade classifiers
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_nose.xml')

# Function to detect and extract facial features
def extract_facial_features(image):
    features = {}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
            # Calculate eye spacing and eye width
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

# Load the processed face image
image = cv2.imread(image_path)
if image is not None:
    # Extract features
    facial_features = extract_facial_features(image)

    # Save features to Excel
    df = pd.DataFrame([facial_features])
    df.to_excel(output_excel_path, index=False)
    print(f"Facial features saved to {output_excel_path}")
else:
    print("Error: Image not found or could not be loaded.")