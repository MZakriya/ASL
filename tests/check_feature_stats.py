import sys
import numpy as np
import os

sys.path.append("d:\\All Projects\\ASLR")
# Use robust stdout
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1, encoding='utf-8', closefd=False)

from main import extract_landmarks_from_video, extract_i3d_features

def check_stats():
    video_path = r"d:\All Projects\ASLR\Video for test\meeting.mp4"
    print(f"Checking stats for {video_path}...")
    
    # 1. MediaPipe Raw
    mp_raw = extract_landmarks_from_video(video_path)
    print(f"MP Raw Shape: {mp_raw.shape}")
    print(f"MP Raw Stats: Mean={mp_raw.mean():.4f}, Std={mp_raw.std():.4f}, Min={mp_raw.min():.4f}, Max={mp_raw.max():.4f}")
    
    # Check if it looks already normalized (e.g. within [-3, 3])
    if mp_raw.max() < 5.0 and mp_raw.min() > -5.0:
        print("WARNING: MP features might ALREADY be normalized!")
    else:
        print("MP features look like raw coordinates (0-1 or pixel).")
        
    # 2. I3D Raw
    try:
        i3d_raw = extract_i3d_features(video_path, target_length=200)
        print(f"I3D Raw Shape: {i3d_raw.shape}")
        print(f"I3D Raw Stats: Mean={i3d_raw.mean():.4f}, Std={i3d_raw.std():.4f}, Min={i3d_raw.min():.4f}, Max={i3d_raw.max():.4f}")
    except Exception as e:
        print(f"I3D Error: {e}")
        
    # 3. Simulate Z-Score
    eps = 1e-6
    mp_norm = (mp_raw - mp_raw.mean(axis=0)) / (mp_raw.std(axis=0) + eps)
    print(f"MP Normalized Stats: Mean={mp_norm.mean():.4f}, Std={mp_norm.std():.4f}")
    
    # Check for dead features (Std < epsilon)
    dead_feats = (mp_raw.std(axis=0) < 1e-6).sum()
    print(f"Dead Features (Std~0): {dead_feats} / {mp_raw.shape[1]}")

check_stats()
