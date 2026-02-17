import os
import sys
import torch
import numpy as np
import pickle

sys.path.append("d:\\All Projects\\ASLR")
from main import extract_landmarks_from_video, extract_i3d_features, Vocabulary
from model import SignLanguageTransformer
from advanced_predict_translation import AdvancedTranslationPredictor

# Unbuffered IO
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1, encoding='utf-8', closefd=False)

def run_ablation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load Resources
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
        
    model = SignLanguageTransformer(vocab_size=9967, d_model=512)
    checkpoint = torch.load("v18_ULTIMATE_POLISHED_E5.pth", map_location=device)
    
    if 'src_proj' in checkpoint: model.src_proj.load_state_dict(checkpoint['src_proj'])
    if 'tgt_emb' in checkpoint: model.tgt_emb.load_state_dict(checkpoint['tgt_emb'])
    if 'fc_out' in checkpoint: model.fc_out.load_state_dict(checkpoint['fc_out'])
    if 'transformer' in checkpoint: model.transformer.load_state_dict(checkpoint['transformer'])
    
    model.to(device)
    model.eval()
    
    # Config
    config = {
        'beam_width': 5,
        'strict_repetition_penalty': 50.0,
        'temperature': 0.4,
        'max_length': 20
    }
    predictor = AdvancedTranslationPredictor(model, vocab, config)
    
    # Process Video
    video_path = r"d:\All Projects\ASLR\Video for test\meeting.mp4"
    if not os.path.exists(video_path):
        print("Video not found.")
        return
        
    print(f"Processing {video_path}...")
    
    # Extract Raw
    mp_raw = extract_landmarks_from_video(video_path)
    if mp_raw.shape[1] > 1629: mp_raw = mp_raw[:, :1629]
    
    try:
        i3d_raw = extract_i3d_features(video_path, target_length=200)
        if i3d_raw.shape[1] > 1024: i3d_raw = i3d_raw[:, :1024]
    except:
        i3d_raw = np.zeros((200, 1024))

    # Interpolate MP to 200
    import torch.nn.functional as F
    mp_t = torch.FloatTensor(mp_raw).unsqueeze(0).transpose(1, 2)
    mp_interp = F.interpolate(mp_t, size=200, mode='linear', align_corners=False)
    mp_final = mp_interp.transpose(1, 2).squeeze(0).numpy()
    i3d_final = i3d_raw
    
    # NORMALIZE (Baseline)
    eps = 1e-6
    mp_norm = (mp_final - mp_final.mean(axis=0)) / (mp_final.std(axis=0) + eps)
    i3d_norm = (i3d_final - i3d_final.mean(axis=0)) / (i3d_final.std(axis=0) + eps)
    
    # EXPERIMENT 1: Baseline (Normalized)
    print("\n--- Exp 1: Baseline (Normalized) ---")
    combo = np.concatenate([mp_norm, i3d_norm], axis=-1)
    batch = torch.from_numpy(combo).float().unsqueeze(0).to(device)
    res = predictor.predict(batch)
    print(f"Result: {res[0]['text']}")
    
    # EXPERIMENT 2: Horizontal Flip MP (x -> 1-x for pose/hands/face)
    print("\n--- Exp 2: Horizontal Flip MP ---")
    # MP structure: Pose(0-99), Face(99-1503), LH(1503-1566), RH(1566-1629)
    # x coordinates are at indices 0, 3, 6...
    mp_flip = mp_norm.copy()
    mp_flip[:, 0::3] = -mp_flip[:, 0::3] # Since it's mean-centered (z-score), flip sign?
    # Wait, simple flip is 1-x on 0-1 data. valid for Z-score?
    # Z-score: (x - u)/s. If x -> 1-x. New mean u' = 1-u. New std same (abs).
    # (1-x - (1-u))/s = (u-x)/s = -(x-u)/s.
    # So YES, flipping sign of Z-scored X-coordinate is equivalent to flipping image!
    
    combo_flip = np.concatenate([mp_flip, i3d_norm], axis=-1)
    batch_flip = torch.from_numpy(combo_flip).float().unsqueeze(0).to(device)
    res_flip = predictor.predict(batch_flip)
    print(f"Result: {res_flip[0]['text']}")
    
    # EXPERIMENT 3: No I3D (Zero out)
    print("\n--- Exp 3: No I3D (MP Only) ---")
    i3d_zero = np.zeros_like(i3d_norm)
    combo_no_i3d = np.concatenate([mp_norm, i3d_zero], axis=-1)
    batch_no_i3d = torch.from_numpy(combo_no_i3d).float().unsqueeze(0).to(device)
    res_no_i3d = predictor.predict(batch_no_i3d)
    print(f"Result: {res_no_i3d[0]['text']}")
    
    # EXPERIMENT 4: I3D Only (Zero MP)
    print("\n--- Exp 4: I3D Only ---")
    mp_zero = np.zeros_like(mp_norm)
    combo_i3d_only = np.concatenate([mp_zero, i3d_norm], axis=-1)
    batch_i3d_only = torch.from_numpy(combo_i3d_only).float().unsqueeze(0).to(device)
    res_i3d_only = predictor.predict(batch_i3d_only)
    print(f"Result: {res_i3d_only[0]['text']}")

    # EXPERIMENT 5: Raw Features (No Extra Norm)
    print("\n--- Exp 5: Raw Features (No Extra Norm) ---")
    combo_raw = np.concatenate([mp_final, i3d_final], axis=-1)
    # Check stats of raw combo
    print(f"Raw Combo Mean: {combo_raw.mean():.4f}, Std: {combo_raw.std():.4f}")
    batch_raw = torch.from_numpy(combo_raw).float().unsqueeze(0).to(device)
    res_raw = predictor.predict(batch_raw)
    print(f"Result: {res_raw[0]['text']}")

if __name__ == "__main__":
    run_ablation()
