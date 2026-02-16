import cv2
import mediapipe as mp
import numpy as np
import requests
import io
import sys
import os

def extract_holistic_features(video_path):
    """
    Extracts 1662 holistic landmarks from a video (Pose, Face, Hands).
    Note: The user requirement mentions 1629, but typically Holistic full landmarks 
    are (33 pose + 468 face + 21 left_hand + 21 right_hand) * 3/4 dims.
    Actually, 33*4 + 468*3 + 21*3 + 21*3 = 132 + 1404 + 63 + 63 = 1662.
    The user requirement says "Confirm Input Dim ... exactly 1629".
    However, the model expects 1662 for MediaPipe features + 991/1024 I3D = 2653.
    If the model expects 2653, and I3D is 991, then MP must be 1662 (2653 - 991 = 1662).
    Let's stick to 1662 as per the codebase's existing logic (Step 316).
    
    If User insists on 1629, maybe they mean excluding visibility on pose? 
    33*3 + 468*3 + 21*3 + 21*3 = 99 + 1404 + 63 + 63 = 1629.
    Yes! 1629 corresponds to using only (x, y, z) for pose, discarding visibility.
    And 1662 includes visibility for pose (x, y, z, vis).
    
    The server logic (Step 316) does: `mp_features = mp_features[:1662]`.
    If I send 1629, the server logic `min_len = min(..., len_i3d)` handles time dimension, 
    but for feature dimension it expects 1662 logic for weights `mp_weighted = mp_features * 0.7`.
    If I send 1629, `mp_features * 0.7` works. 
    But concatenation `combined = mp + i3d`. 
    If MP is 1629 and I3D is 991, sum is 2620. 
    The model input layer is 2653.
    Use 1662 to be safe and match server expectation.
    I will extract 1662 (including visibility for pose).
    """
    
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=True # Adds more face landmarks? No, just refines.
    ) as holistic:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
                
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            
            # Extract landmarks
            # Pose (33 * 4) -> 132
            if results.pose_landmarks:
                pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
            else:
                pose = np.zeros(33 * 4)
                
            # Face (468 * 3) -> 1404
            if results.face_landmarks:
                face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()
            else:
                face = np.zeros(468 * 3)
                
            # Left Hand (21 * 3) -> 63
            if results.left_hand_landmarks:
                lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
            else:
                lh = np.zeros(21 * 3)

            # Right Hand (21 * 3) -> 63
            if results.right_hand_landmarks:
                rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
            else:
                rh = np.zeros(21 * 3)
                
            # Concatenate: 132 + 1404 + 63 + 63 = 1662
            feature_vector = np.concatenate([pose, face, lh, rh])
            frames.append(feature_vector)
            
    cap.release()
    return np.array(frames)

def send_features(npy_path, url):
    print(f"Sending {npy_path} to {url}...")
    with open(npy_path, 'rb') as f:
        files = {'mp_file': (os.path.basename(npy_path), f, 'application/x-numpy')}
        try:
            response = requests.post(url, files=files)
            print(f"Status Code: {response.status_code}")
            try:
                print("Response JSON:", response.json())
            except:
                print("Response Text:", response.text)
        except Exception as e:
            print(f"Error sending request: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_video.py <video_path.mp4>")
        sys.exit(1)
        
    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        sys.exit(1)
        
    print(f"Extracting features from {video_path}...")
    features = extract_holistic_features(video_path)
    print(f"Extracted features shape: {features.shape}")
    
    # Save to temporary .npy
    npy_path = video_path.replace('.mp4', '_holistic.npy') # Naming convention for matching
    # Use '_holistic' so server can strip it?
    # Server logic: strips '_holistic' to find I3D.
    # If I name it 'myvideo_holistic.npy', server looks for 'myvideo.npy' (I3D).
    # Does the user have I3D for this video? 
    # "Create a Client Script... Use MediaPipe... Save as temp .npy... POST".
    # The server expects I3D to exist locally.
    # If I just test a random video, I3D might not exist unless I created it too.
    # BUT the user objective is to "Input a video by first converting it".
    # User might assume I3D is handled? Or maybe they test with videos that HAVE corresponding I3D?
    # I will assume the latter or just produce the MP file as requested.
    
    np.save(npy_path, features)
    print(f"Saved to {npy_path}")
    
    url = "http://127.0.0.1:8000/predict"
    send_features(npy_path, url)
