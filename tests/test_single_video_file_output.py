import os
import sys
import torch
import numpy as np
import cv2
import mediapipe as mp
import time

# Add project root
sys.path.append("d:\\All Projects\\ASLR")

# Import necessary modules
from main import extract_landmarks_from_video, extract_i3d_features, Vocabulary, AdvancedTranslationPredictor, model as main_model, device, vocab as main_vocab, predictor
# We need to manually load model/vocab if main.py doesn't do it on import (it does it in lifespan)
from model import SignLanguageTransformer
import pickle

def load_resources():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Vocab
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
        
    # Verify itos
    if not hasattr(vocab, 'itos') and hasattr(vocab, 'index2word'):
        vocab.itos = vocab.index2word
        
    # Load Model
    model = SignLanguageTransformer(vocab_size=9967, d_model=512)
    checkpoint = torch.load("v18_ULTIMATE_POLISHED_E5.pth", map_location=device)
    
    # Load Weights manually as per main.py logic
    if 'transformer' in checkpoint: model.transformer.load_state_dict(checkpoint['transformer'])
    if 'src_proj' in checkpoint: model.src_proj.load_state_dict(checkpoint['src_proj'])
    if 'tgt_emb' in checkpoint: model.tgt_emb.load_state_dict(checkpoint['tgt_emb'])
    if 'fc_out' in checkpoint: model.fc_out.load_state_dict(checkpoint['fc_out'])
    
    model.to(device)
    model.eval()
    
    return model, vocab, device

def process_video(video_path, model, vocab, device):
    # 1. Extract MP
    # extract_landmarks_from_video prints to stdout, so we might not see it, but it should work
    mp_features = extract_landmarks_from_video(video_path)
    if len(mp_features) == 0:
        return "ERROR: No landmarks"
        
    if mp_features.shape[1] > 1629: mp_features = mp_features[:, :1629]
    
    # 2. Extract I3D (Force 200 frames)
    # We need to be careful about not re-initializing I3D every time
    try:
        i3d_features = extract_i3d_features(video_path, target_length=200)
    except Exception as e:
        i3d_features = np.zeros((200, 1024)) # Fallback
        
    # 3. Interpolate MP
    TARGET_FRAMES = 200
    import torch.nn.functional as F
    mp_t = torch.FloatTensor(mp_features).unsqueeze(0).transpose(1, 2)
    mp_interp = F.interpolate(mp_t, size=TARGET_FRAMES, mode='linear', align_corners=False)
    mp_final = mp_interp.transpose(1, 2).squeeze(0).numpy()
    
    # 4. Handle I3D Shape
    i3d_final = i3d_features
    if i3d_final.shape[1] > 1024: i3d_final = i3d_final[:, :1024]
    
    # 5. NORMALIZE (The Fix)
    eps = 1e-6
    mp_final = (mp_final - mp_final.mean(axis=0)) / (mp_final.std(axis=0) + eps)
    i3d_final = (i3d_final - i3d_final.mean(axis=0)) / (i3d_final.std(axis=0) + eps)
    
    # 6. Concat
    mp_tensor = torch.from_numpy(mp_final)
    i3d_tensor = torch.from_numpy(i3d_final)
    combined = torch.cat([mp_tensor, i3d_tensor], dim=-1).float().to(device)
    
    input_batch = combined.unsqueeze(0)
    
    # 7. Predict
    from advanced_predict_translation import AdvancedTranslationPredictor
    config = {
        'beam_width': 5, # Beam Search
        'strict_repetition_penalty': 50.0, # Maximized penalty
        'temperature': 0.4,
        'max_length': 20
    }
    predictor = AdvancedTranslationPredictor(model, vocab, config)
    
    try:
        with open("debug_process.log", "a") as f: f.write(f"START PROCESSING: {video_path}\n")
        
        # Test 1: Real Video
        print(f"--- Predicting on {video_path} ---")
        with open("debug_process.log", "a") as f: f.write("Predicting on Real Video...\n")
        results = predictor.predict(input_batch)
        pred_text = results[0]['text']
        with open("debug_process.log", "a") as f: f.write(f"Result: {pred_text}\n")
        
        # Test 2: Zero Input (Blindness Check)
        print("--- Predicting on ZEROS ---")
        with open("debug_process.log", "a") as f: f.write("Predicting on ZEROS...\n")
        zeros = torch.zeros_like(input_batch)
        results_zeros = predictor.predict(zeros)
        pred_zeros = results_zeros[0]['text']
        with open("debug_process.log", "a") as f: f.write(f"Result Zeros: {pred_zeros}\n")
        
        return f"Video: {pred_text}\nZeros: {pred_zeros}"
    except Exception as e:
        with open("debug_process.log", "a") as f: f.write(f"ERROR: {e}\n")
        return f"Prediction Error: {e}"

def main():
    try:
        test_file = r"d:\All Projects\ASLR\Video for test\meeting.mp4"
        if not os.path.exists(test_file):
             # Try locate any mp4
             test_dir = r"d:\All Projects\ASLR\Video for test"
             files = [f for f in os.listdir(test_dir) if f.endswith('.mp4')]
             if files:
                 test_file = os.path.join(test_dir, files[0])
             else:
                 with open("tests/result.txt", "w") as f:
                     f.write("ERROR: No test video found")
                 return
                 
        model, vocab, device = load_resources()
        
        prediction = process_video(test_file, model, vocab, device)
        
        with open("tests/result.txt", "w") as f:
            f.write(f"FILE: {test_file}\nPrediction: {prediction}")
            
    except Exception as e:
        with open("tests/result.txt", "w") as f:
            f.write(f"CRITICAL EXCEPTION: {e}")

if __name__ == "__main__":
    main()
