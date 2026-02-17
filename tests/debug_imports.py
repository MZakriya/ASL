import sys
import os

with open("tests/debug_imports.log", "w") as f:
    f.write("Start\n")
    f.flush()
    
    try:
        import torch
        f.write("Imported torch\n")
        f.flush()
        
        import cv2
        f.write("Imported cv2\n")
        f.flush()
        
        import mediapipe
        f.write("Imported mediapipe\n")
        f.flush()
        
        sys.path.append("d:\\All Projects\\ASLR")
        from main import extract_landmarks_from_video
        f.write("Imported main\n")
        f.flush()
        
    except Exception as e:
        f.write(f"Error: {e}\n")
