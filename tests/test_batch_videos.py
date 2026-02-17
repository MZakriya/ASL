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
    print("Loading resources manually...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Vocab
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    print(f"Vocab loaded. Size: {len(vocab)}")
    
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
    print(f"\nProcessing {os.path.basename(video_path)}...")
    
    # 1. Extract MP
    mp_features = extract_landmarks_from_video(video_path)
    if len(mp_features) == 0:
        return "ERROR: No landmarks"
        
    if mp_features.shape[1] > 1629: mp_features = mp_features[:, :1629]
    
    # 2. Extract I3D (Force 200 frames)
    # We need to be careful about not re-initializing I3D every time if possible, but for test script it's fine
    try:
        i3d_features = extract_i3d_features(video_path, target_length=200)
    except Exception as e:
        print(f"I3D Error: {e}")
        i3d_features = np.zeros((200, 1024)) # Fallback
        
    # 3. Interpolate MP
    TARGET_FRAMES = 200
    import torch.nn.functional as F
    mp_t = torch.FloatTensor(mp_features).unsqueeze(0).transpose(1, 2)
    mp_interp = F.interpolate(mp_t, size=TARGET_FRAMES, mode='linear', align_corners=False)
    mp_final = mp_interp.transpose(1, 2).squeeze(0).numpy()
    
    # 4. Handle I3D Shape
    i3d_final = i3d_features
    if i3d_final.shape[0] != TARGET_FRAMES:
         # simple reshape/pad logic (main.py handles this better but let's assume it worked)
         pass
    
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
    # Initialize predictor wrapper
    from advanced_predict_translation import AdvancedTranslationPredictor
    config = {
        'beam_width': 1,
        'strict_repetition_penalty': 5.0,
        'temperature': 0.4,
        'max_length': 12
    }
    predictor = AdvancedTranslationPredictor(model, vocab, config)
    
    try:
        results = predictor.predict(input_batch)
        return results[0]['text']
    except Exception as e:
        return f"Prediction Error: {e}"

def main():
    test_dir = r"d:\All Projects\ASLR\Video for test"
    if not os.path.exists(test_dir):
        print(f"Directory not found: {test_dir}")
        return
        
    model, vocab, device = load_resources()
    
    print(f"\nScanning {test_dir}...")
    videos = [f for f in os.listdir(test_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    print(f"Found {len(videos)} videos.")
    
    for video_file in videos:
        full_path = os.path.join(test_dir, video_file)
        prediction = process_video(full_path, model, vocab, device)
        print(f"RESULT [{video_file}]: {prediction}")

if __name__ == "__main__":
    main()
