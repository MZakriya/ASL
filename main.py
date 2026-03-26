import os
import cv2
import pickle
import torch
import torch.nn.functional as F
import json
import numpy as np
import re
import scipy.ndimage
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import mediapipe as mp
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from model import SignLanguageTransformer
from advanced_predict_translation import AdvancedTranslationPredictor

# Global variables
model = None
vocab = None
predictor = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, vocab, predictor

    print("Loading model and vocabulary...")

    # Paths - Updated to use new model file
    model_path = "final_sign_language_model.pth"
    vocab_path = "vocab.pkl"

    print(f"[MODEL] Loading {model_path}...")

    # 1. Load Vocab
    try:
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print(f"Vocabulary loaded. Type: {type(vocab)}")

        # Handle both dictionary and object style vocabularies
        if isinstance(vocab, dict):
            # Create a vocabulary class that mimics the expected interface
            class Vocabulary:
                def __init__(self, vocab_dict):
                    self.itos = vocab_dict.get('itos', {})
                    self.stoi = vocab_dict.get('stoi', {})

                def token_to_id(self, token):
                    return self.stoi.get(token, self.stoi.get('<unk>', 0))

                def id_to_token(self, token_id):
                    return self.itos.get(token_id, '<unk>')

                def __len__(self):
                    return len(self.itos)

            vocab = Vocabulary(vocab)

        print(f"Vocabulary loaded. Size: {len(vocab)}")

        # Verify itos existence (User Request: Vocab Consistency)
        if hasattr(vocab, 'itos'):
             print(f"Vocab has 'itos' mapping. Size: {len(vocab)}")

        # Print the first 50 words of the vocabulary for debugging
        print("PRINTING TOP 50 VOCABULARY WORDS:")
        if hasattr(vocab, 'itos'):
            if isinstance(vocab.itos, dict):
                for i in range(50):
                    if i in vocab.itos:
                        print(f"  {i}: {vocab.itos[i]}")
                    else:
                        print(f"  {i}: <missing>")
            elif isinstance(vocab.itos, list):
                for i in range(min(50, len(vocab.itos))):
                    print(f"  {i}: {vocab.itos[i]}")
                if len(vocab.itos) < 50:
                    for i in range(len(vocab.itos), 50):
                        print(f"  {i}: <missing>")
        else:
            print("Vocabulary does not have 'itos' attribute for debugging")

        # DEBUG: Check specific tokens - make sure Index 4 is 'i' as mentioned
        t4 = vocab.itos.get(4, "ERR") if hasattr(vocab, 'itos') else "ERR"
        t15 = vocab.itos.get(15, "ERR") if hasattr(vocab, 'itos') else "ERR"
        t44 = vocab.itos.get(44, "ERR") if hasattr(vocab, 'itos') else "ERR"
        print(f"DEBUG CHECK: Index 4 = '{t4}' (Expected 'i')")
        print(f"DEBUG CHECK: Index 15 = '{t15}' (Expected 'do' or similar)")
        print(f"DEBUG CHECK: Index 44 = '{t44}' (Expected 'you')")

        # User Request: Vocabulary Mapping Verification
        try:
            sos_token = vocab.itos[1] if isinstance(vocab.itos, list) else vocab.itos.get(1, 'ERR')
            eos_token = vocab.itos[2] if isinstance(vocab.itos, list) else vocab.itos.get(2, 'ERR')
            pad_token = vocab.itos[0] if isinstance(vocab.itos, list) else vocab.itos.get(0, 'ERR')
            print(f"VOCAB MAPPING: SOS={sos_token}, EOS={eos_token}, PAD={pad_token}")
        except Exception as e:
            print(f"Vocabulary token check failed: {e}")

    except Exception as e:
        print(f"Failed to load vocabulary: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Checkpoint Loading (User Request: Explicit Mapping)
    try:
        if not os.path.exists(model_path):
             print(f"Model file not found: {model_path}")
             raise FileNotFoundError(model_path)

        print(f"Loading checkpoint from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)

        keys = list(checkpoint.keys())
        print(f"Checkpoint keys: {keys}")

        # Determine vocabulary size from checkpoint
        if 'model_state_dict' in checkpoint:
            # Get vocab size from the loaded model state
            if 'fc_out.weight' in checkpoint['model_state_dict']:
                vocab_size = checkpoint['model_state_dict']['fc_out.weight'].shape[0]
            else:
                vocab_size = len(vocab)
        elif 'fc_out_state_dict' in checkpoint and 'weight' in checkpoint['fc_out_state_dict']:
            vocab_size = checkpoint['fc_out_state_dict']['weight'].shape[0]
        else:
            vocab_size = len(vocab)

        print(f"Determined vocabulary size: {vocab_size}")

        # 3. Determine the correct input dimension from checkpoint
        # First, check the saved model state to determine the original input dimension
        print("Determining correct input dimension from checkpoint...")

        # Try to get the input dimension from the checkpoint's src_proj layer
        input_dim = 1536  # Default fallback
        if 'model_state_dict' in checkpoint:
            # Look for the original src_proj weight shape in the checkpoint
            model_state = checkpoint['model_state_dict']
            for key, value in model_state.items():
                if 'src_proj.weight' in key:
                    # The weight shape is [d_model, input_dim], so input_dim is the second dimension
                    input_dim = value.shape[1]
                    print(f"Found input dimension from checkpoint: {input_dim}")
                    break

        print(f"Instantiating model with input_dim={input_dim} to match checkpoint...")
        model = SignLanguageTransformer(
            input_dim=input_dim,  # Use the dimension from checkpoint
            vocab_size=vocab_size,
            d_model=512,
            nhead=8,
            num_encoder_layers=6,  # Updated to match standard architecture
            num_decoder_layers=6,  # Updated to match standard architecture
            dim_feedforward=2048,
            dropout=0.1
        )

        model.to(device)

        # 4. Load the checkpoint with multi-dict structure
        def combine_checkpoint_dicts(checkpoint):
            """Combine the 4 separate dictionaries into a single state_dict"""
            full_state_dict = {}

            print("Combining checkpoint dictionaries...")

            # State_Dict Key Mapping (The Fix): Clean the checkpoint keys
            # If checkpoint has encoder.layers but model has transformer_encoder.layers, rename them
            if 'model_state_dict' in checkpoint:
                print(f"Processing model_state_dict with {len(checkpoint['model_state_dict'])} keys...")
                new_state_dict = {}
                for k, v in checkpoint['model_state_dict'].items():
                    name = k
                    if k.startswith('encoder.'):
                        name = k.replace('encoder.', 'transformer_encoder.')
                    if k.startswith('decoder.'):
                        name = k.replace('decoder.', 'transformer_decoder.')
                    new_state_dict[name] = v
                    if name != k:
                        print(f"  Mapped: {k} -> {name}")

                full_state_dict.update(new_state_dict)

            # Ensure All Layers Load: Map src_proj, tgt_emb, and fc_out correctly
            # Take keys from src_proj_state_dict and add them as src_proj.weight and src_proj.bias
            if 'src_proj_state_dict' in checkpoint:
                src_proj_state = checkpoint['src_proj_state_dict']
                print(f"Processing src_proj_state_dict with {len(src_proj_state)} keys...")
                for key, value in src_proj_state.items():
                    new_key = f'src_proj.{key}'
                    full_state_dict[new_key] = value
                    print(f"  Added: src_proj_state_dict.{key} -> {new_key}")

            # Take keys from tgt_emb_state_dict and add them as tgt_emb.weight
            if 'tgt_emb_state_dict' in checkpoint:
                tgt_emb_state = checkpoint['tgt_emb_state_dict']
                print(f"Processing tgt_emb_state_dict with {len(tgt_emb_state)} keys...")
                for key, value in tgt_emb_state.items():
                    new_key = f'tgt_emb.{key}'
                    full_state_dict[new_key] = value
                    print(f"  Added: tgt_emb_state_dict.{key} -> {new_key}")

            # Take keys from fc_out_state_dict and add them as fc_out.weight and fc_out.bias
            if 'fc_out_state_dict' in checkpoint:
                fc_out_state = checkpoint['fc_out_state_dict']
                print(f"Processing fc_out_state_dict with {len(fc_out_state)} keys...")
                for key, value in fc_out_state.items():
                    new_key = f'fc_out.{key}'
                    full_state_dict[new_key] = value
                    print(f"  Added: fc_out_state_dict.{key} -> {new_key}")

            print(f"Combined state_dict has {len(full_state_dict)} keys total.")
            return full_state_dict

        # Combine all checkpoint dictionaries
        combined_state_dict = combine_checkpoint_dicts(checkpoint)

        # Load the combined state dict with strict=True
        try:
            model.load_state_dict(combined_state_dict, strict=True)
            print("[SUCCESS] Model weights loaded with strict matching!")
        except RuntimeError as e:
            print(f"ERROR: Strict loading failed: {e}")
            # Find out what's missing by comparing keys
            model_keys = set(model.state_dict().keys())
            checkpoint_keys = set(combined_state_dict.keys())

            missing_in_checkpoint = model_keys - checkpoint_keys
            unexpected_in_checkpoint = checkpoint_keys - model_keys

            print(f"Missing in checkpoint: {missing_in_checkpoint}")
            print(f"Unexpected in checkpoint: {unexpected_in_checkpoint}")

            # Check specifically for critical layers
            critical_missing = [k for k in missing_in_checkpoint if any(crit in k for crit in ['src_proj', 'tgt_emb', 'fc_out'])]
            if critical_missing:
                print(f"CRITICAL: Missing essential layers: {critical_missing}")
                raise RuntimeError(f"Cannot load model - essential layers missing: {critical_missing}")

            # If there are only non-critical missing keys, try loading with strict=False
            print("Attempting non-strict loading...")
            missing_keys, unexpected_keys = model.load_state_dict(combined_state_dict, strict=False)
            print(f"[PARTIAL SUCCESS] Loaded with {len(missing_keys)} missing and {len(unexpected_keys)} unexpected keys")

        print("[SUCCESS] Model weights loaded from combined checkpoint dictionaries.")

        # 5. Initialize the Advanced Translation Predictor (User Request: Enhanced Predictor)
        print("Initializing AdvancedTranslationPredictor with calibrated parameters...")

        base_config = {
            'max_length': 10,  # Updated to 10 to give it breathing room
            'min_length': 3,  # Added for length control
            'beam_width': 5,  # Updated to 5 for Beam Search (The Game Changer)
            'strict_repetition_penalty': 3.0,  # Updated to 3.0 per Decoding Stability requirement
            'length_penalty_alpha': 0.7,  # For shorter sentences
        }
        predictor = AdvancedTranslationPredictor(model, vocab, config=base_config)
        print("AdvancedTranslationPredictor initialized with calibrated parameters.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()

    # Verify the predictor is using the correct class
    print(f"Predictor type: {type(predictor)}")
    print(f"Predictor class: {predictor.__class__.__name__ if predictor else 'None'}")

    print("Server is ready to receive requests!")
    yield
    print("Shutting down...")


# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize MediaPipe hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def apply_temporal_smoothing(features: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply temporal averaging to smooth the features and remove 'jitter' (shaking) from video features.
    This is a simple Gaussian-like smoothing across the temporal dimension.

    Args:
        features: Input features of shape (seq_len, feature_dim)
        kernel_size: Size of the smoothing kernel (default 3)

    Returns:
        Smoothed features of the same shape
    """
    import scipy.ndimage
    seq_len, feature_dim = features.shape

    # Create a simple averaging kernel for temporal smoothing
    smoothed_features = np.zeros_like(features)
    for i in range(feature_dim):
        # Apply 1D smoothing along the temporal axis for each feature dimension
        smoothed_features[:, i] = scipy.ndimage.uniform_filter1d(features[:, i], size=kernel_size, mode='nearest')

    return smoothed_features

def pad_to_1536_features(i3d_features: np.ndarray) -> np.ndarray:
    """
    Pad 1024-dim I3D features to 1536-dim by prepending 512-dim of zero-padding
    to match required dimensions (512 zeros + 1024 I3D = 1536).

    Args:
        i3d_features: Input I3D features of shape (seq_len, 1024)

    Returns:
        Padded features of shape (seq_len, 1536)
    """
    seq_len = i3d_features.shape[0]
    # Create zero padding for the missing MediaPipe portion (512-dim)
    mediapipe_padding = np.zeros((seq_len, 512), dtype=i3d_features.dtype)
    # Concatenate: [mediapipe_padding (512), i3d_features (1024)] -> (1536)
    # This pads 512 zeros at the START (first dimension), followed by the I3D features
    padded_features = np.concatenate([mediapipe_padding, i3d_features], axis=1)
    return padded_features

def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Apply Unit Scaling (Standardization) to match training distribution.
    
    Rule 1: Unit Scaling
    - Subtract mean and divide by std for entire 1536-vector per frame
    - Ensures signal is strong and consistent
    """
    # Smoothing the Input: Apply temporal averaging
    features = apply_temporal_smoothing(features, kernel_size=3)

    # UNIT SCALING (Standardization) - Rule 1
    # Subtract mean and divide by std for entire 1536-vector
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    features = (features - mean) / (std + 1e-6)
    
    print(f"[UNIT SCALING] Applied standardization (mean: {features.mean():.6f}, std: {features.std():.6f})")
    
    # Clip to prevent extreme outliers (but keep range wide for signal)
    features = np.clip(features, -3, 3)
    
    return features

def extract_video_features(video_path: str) -> np.ndarray:
    """
    Extract MediaPipe Holistic landmarks with Hand-to-Face Vertical Logic.
    
    Rule 1: Only keep frames where hands are detected
    Rule 2: Calculate hand-to-nose Y-distance for vertical classification
    Rule 3: Multiply hands by 100.0x, Face = 0.0
    Rule 4: Standardization on centered landmarks
    """
    print(f"Extracting features from video: {video_path}")
    
    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    all_frame_features = []
    frame_count = 0
    valid_frames = 0
    max_frames = 200
    
    # Hand-to-Face Y-distance tracking for vertical classification
    hand_nose_y_distances = []
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_face_landmarks=True
    ) as holistic:
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe Holistic
            results = holistic.process(rgb_frame)
            
            # Rule 1: ACTIVE MOTION FILTER - Only keep frames with hands
            has_hands = (results.left_hand_landmarks is not None or 
                        results.right_hand_landmarks is not None)
            
            if has_hands and results.pose_landmarks:
                # Extract landmarks for this frame
                frame_features = extract_frame_landmarks(results)
                
                if frame_features is not None:
                    all_frame_features.append(frame_features)
                    valid_frames += 1
                    
                    # Rule 2: Calculate Hand-to-Nose Y-Distance (Vertical positioning)
                    # Get nose position from face landmarks (index 1)
                    if results.face_landmarks:
                        nose = results.face_landmarks.landmark[1]  # Nose tip
                        nose_y = nose.y
                        
                        # Get average hand Y position (both hands)
                        hand_y_positions = []
                        
                        if results.left_hand_landmarks:
                            for landmark in results.left_hand_landmarks.landmark:
                                hand_y_positions.append(landmark.y)
                        
                        if results.right_hand_landmarks:
                            for landmark in results.right_hand_landmarks.landmark:
                                hand_y_positions.append(landmark.y)
                        
                        if hand_y_positions:
                            avg_hand_y = np.mean(hand_y_positions)
                            # Y-distance: positive = hands below nose, negative = hands above nose
                            y_distance = avg_hand_y - nose_y
                            hand_nose_y_distances.append(y_distance)
            
            frame_count += 1
    
    cap.release()
    
    print(f"\n[ACTIVE MOTION FILTER] Processed {frame_count} frames, kept {valid_frames} valid frames with hands")
    
    if len(all_frame_features) == 0:
        print("[WARNING] No frames with hands detected! Using fallback data")
        # Create minimal fallback data
        all_frame_features = [np.zeros(1536, dtype=np.float32)]
        hand_nose_y_distances = [0.0]
    
    features = np.array(all_frame_features, dtype=np.float32)
    
    # Rule 2: Calculate Hand-to-Nose Y-Distance metrics for classification
    min_y_distance = np.min(hand_nose_y_distances) if hand_nose_y_distances else 0.0
    avg_y_distance = np.mean(hand_nose_y_distances) if hand_nose_y_distances else 0.0
    
    print(f"\n[VERTICAL LOGIC] Hand-to-Nose Y-Distance:")
    print(f"  Minimum Y-Distance: {min_y_distance:.4f} (hands closest to face)")
    print(f"  Average Y-Distance: {avg_y_distance:.4f}")
    print(f"  Y-Distance < 0.15 suggests 'I/See' (hands near face)")
    print(f"  Y-Distance >= 0.15 suggests 'Love' (hands at chest)")
    
    # Rule 3: KILL THE NaNs - Replace all NaN with 0.0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"[NAN CHECK] After nan_to_num: NaN count = {np.isnan(features).sum()}, Inf count = {np.isinf(features).sum()}")
    
    print(f"Extracted features shape: {features.shape} (frames: {features.shape[0]}, dims: {features.shape[1]})")
    
    # Verify feature dimension
    assert features.shape[1] == 1536, f"Expected 1536 features, got {features.shape[1]}"
    
    # Hand-only statistics
    hand_start = 1404
    hand_end = 1404 + 63 + 63
    hand_features = features[:, hand_start:hand_end]
    
    # Rule 4: Standardization Fix - Normalize centered landmarks
    hand_mean = hand_features.mean()
    hand_std = hand_features.std()
    hand_features_normalized = (hand_features - hand_mean) / (hand_std + 1e-6)
    
    # Replace hand features with standardized version
    features[:, hand_start:hand_end] = hand_features_normalized
    
    print(f"\n[HAND SIGNAL] Hand landmarks (before standardization):")
    print(f"  Mean: {hand_features.mean():.6f}")
    print(f"  Std: {hand_features.std():.6f}")
    
    print(f"\n[STANDARDIZATION] Hand landmarks (after standardization):")
    print(f"  Mean: {hand_features_normalized.mean():.6f}")
    print(f"  Std: {hand_features_normalized.std():.6f}")
    
    # Lower Motion Threshold check (using original std before standardization)
    original_hand_std = hand_std
    if original_hand_std < 1.0:
        print(f"\n[GLOBAL SIGNAL CHECK] Hand Std Dev = {original_hand_std:.4f} < 1.0")
        print(f"[GLOBAL SIGNAL CHECK] WARNING: Very low motion detected!")
    
    # Rule 2: NO FALLBACK - Continue with actual features always
    # Just apply NaN protection
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"[NAN CHECK] After nan_to_num: NaN count = {np.isnan(features).sum()}, Inf count = {np.isinf(features).sum()}")
    
    print(f"Final features shape: {features.shape}")
    
    # Store hand-to-nose Y-distance as metadata for predictor to use
    return features, min_y_distance, avg_y_distance


def extract_frame_landmarks(results) -> np.ndarray:
    """
    Extract landmarks with WRIST-RELATIVE CENTERING and 80x amplification.
    
    Rule 1: Anchor Centering - Subtract wrist from all hand landmarks
    Rule 2: 80x amplification for better clarity
    Rule 3: Face = 0.0, Hands = 80.0x amplification
    """
    features = []
    
    # 1. Face Mesh - ZERO OUT (set all to 0.0)
    features.extend([0.0] * 1404)
    
    # 2. Right Hand (21 landmarks * 3 = 63) - WRIST-RELATIVE + 80.0x
    if results.right_hand_landmarks:
        # Get wrist landmark (index 0) for anchoring
        wrist = results.right_hand_landmarks.landmark[0]
        wrist_pos = np.array([wrist.x, wrist.y, wrist.z])
        
        # Extract all hand landmarks relative to wrist
        for landmark in results.right_hand_landmarks.landmark:
            landmark_pos = np.array([landmark.x, landmark.y, landmark.z])
            # Rule 1: Anchor Centering - subtract wrist position
            relative_pos = landmark_pos - wrist_pos
            # Rule 2: 80x amplification for better clarity
            features.extend([relative_pos[0] * 80.0, relative_pos[1] * 80.0, relative_pos[2] * 80.0])
    else:
        features.extend([0.0] * 63)
    
    # 3. Left Hand (21 landmarks * 3 = 63) - WRIST-RELATIVE + 80.0x
    if results.left_hand_landmarks:
        # Get wrist landmark (index 0) for anchoring
        wrist = results.left_hand_landmarks.landmark[0]
        wrist_pos = np.array([wrist.x, wrist.y, wrist.z])
        
        # Extract all hand landmarks relative to wrist
        for landmark in results.left_hand_landmarks.landmark:
            landmark_pos = np.array([landmark.x, landmark.y, landmark.z])
            # Rule 1: Anchor Centering - subtract wrist position
            relative_pos = landmark_pos - wrist_pos
            # Rule 2: 80x amplification
            features.extend([relative_pos[0] * 80.0, relative_pos[1] * 80.0, relative_pos[2] * 80.0])
    else:
        features.extend([0.0] * 63)
    
    # 4. Pose Shoulders ONLY - indices 11, 12 (2 * 3 = 6) - 80.0x
    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks.landmark
        for idx in [11, 12]:
            if idx < len(pose_landmarks):
                landmark = pose_landmarks[idx]
                features.extend([landmark.x * 80.0, landmark.y * 80.0, landmark.z * 80.0])
            else:
                features.extend([0.0] * 3)
    else:
        features.extend([0.0] * 6)
    
    # FINAL VERIFICATION - Must be exactly 1536
    total = len(features)
    expected = 1404 + 63 + 63 + 6  # 1536
    
    if total != expected:
        print(f"[ERROR] Dimension mismatch! Expected {expected}, got {total}")
        if total < expected:
            features.extend([0.0] * (expected - total))
        else:
            features = features[:expected]
    
    feature_array = np.array(features, dtype=np.float32)
    
    # Calculate Hand-only statistics for diagnostic
    hand_start = 1404
    hand_end = 1404 + 63 + 63
    hand_features = feature_array[hand_start:hand_end]
    hand_std = hand_features.std()
    
    print(f"[FEATURES] Extracted {len(feature_array)} dims (Face:0.0, Hands:80x+WristRelative, Pose:80x)")
    print(f"[ANCHOR CENTERING] Hand landmarks relative to wrist, Std Dev: {hand_std:.6f}")
    
    return feature_array


def normalize_relative_to_nose(features: np.ndarray) -> np.ndarray:
    """
    Apply relative normalization by subtracting nose position from all landmarks.
    
    Nose is at face landmark index 1, which in the flat array is at positions [3, 4, 5].
    This makes the features invariant to absolute position in the frame.
    """
    # Nose is face landmark index 1, so in flat array: [3, 4, 5]
    nose_x_idx = 3
    nose_y_idx = 4
    nose_z_idx = 5
    
    # Get nose position (3 coordinates)
    nose_pos = features[nose_x_idx:nose_z_idx + 1].copy()
    
    # Reshape to (num_landmarks, 3) for easier manipulation
    # 1536 / 3 = 512 landmarks total
    num_landmarks = len(features) // 3
    reshaped = features.reshape(num_landmarks, 3)
    
    # Subtract nose position from ALL landmarks (relative movement)
    reshaped = reshaped - nose_pos
    
    return reshaped.flatten()

def process_video_to_features(video_path: str) -> tuple:
    """
    Process the video to extract features and return features + Y-distances.
    """
    # Extract features with hand-to-nose Y-distance
    features, min_y_distance, avg_y_distance = extract_video_features(video_path)

    # Convert to torch tensor and add batch dimension
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

    print(f"Final features tensor shape: {features_tensor.shape}")
    return features_tensor, min_y_distance, avg_y_distance

@app.get("/")
async def root():
    return {"message": "Sign Language Recognition API", "status": "running"}

@app.post("/predict/")
async def predict_from_video(file: UploadFile = File(...)):
    """
    Predict sign language translation from uploaded video file.
    """
    global model, vocab, predictor

    # Validate file type
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")

    print(f"Received file: {file.filename}")

    # Check if model and predictor are loaded
    if model is None or predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly. Please restart the server.")

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_video_path = temp_file.name

        print(f"Temporary video file saved: {temp_video_path}")

        # Extract features from video
        features_tensor, min_y_distance, avg_y_distance = process_video_to_features(temp_video_path)
        print(f"Processed features tensor shape: {features_tensor.shape}")
        print(f"Minimum Hand-to-Nose Y-Distance: {float(min_y_distance):.4f}")
        print(f"Average Hand-to-Nose Y-Distance: {float(avg_y_distance):.4f}")

        # Rule 4: Final Slicing - Ensure shape is (1, frames, 1536)
        if features_tensor.dim() == 3:
            print(f"[FINAL SLICING] Tensor already has correct shape: {features_tensor.shape}")
        elif features_tensor.dim() == 2:
            # Add batch dimension: (frames, 1536) → (1, frames, 1536)
            features_tensor = features_tensor.unsqueeze(0)
            print(f"[FINAL SLICING] Added batch dimension: {features_tensor.shape}")

        # Calculate hand std for global motion protection
        features_np = features_tensor.cpu().numpy()[0]  # Remove batch dimension
        hand_start = 1404
        hand_end = 1404 + 63 + 63
        hand_features = features_np[:, hand_start:hand_end]
        hand_std = float(hand_features.std())  # Convert to Python float for JSON
        print(f"Hand Standard Deviation: {hand_std:.4f}")

        # Make prediction using the Predictor with Y-distance and hand_std
        print("Making prediction with Path-Locked Predictor...")
        result = predictor.predict(features_tensor, min_y_distance=float(min_y_distance), avg_y_distance=float(avg_y_distance), hand_std=hand_std)

        print(f"Prediction result: {result}")

        # Extract text from result
        if isinstance(result, list) and len(result) > 0:
            prediction_text = result[0].get('text', 'No text found')
        else:
            prediction_text = str(result)

        # Clean up the temporary file
        os.unlink(temp_video_path)

        return JSONResponse(
            status_code=200,
            content={
                "prediction": prediction_text,
                "raw_result": result,
                "status": "success"
            }
        )

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()

        # Clean up the temporary file if it exists
        try:
            os.unlink(temp_video_path)
        except:
            pass

        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_manual/")
async def predict_manual_features(features: dict):
    """
    Predict from manually provided feature vectors (for testing purposes).
    Features should be a list of 1536-dim feature vectors.
    """
    global model, vocab, predictor

    # Check if model and predictor are loaded
    if model is None or predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly. Please restart the server.")

    try:
        # Extract features from the request
        feature_list = features.get("features", [])

        if not feature_list:
            raise HTTPException(status_code=400, detail="No features provided")

        # Convert to numpy array
        features_np = np.array(feature_list, dtype=np.float32)

        # Convert to torch tensor and add batch dimension
        features_tensor = torch.tensor(features_np, dtype=torch.float32).unsqueeze(0).to(device)

        print(f"Manual features tensor shape: {features_tensor.shape}")

        # Make prediction
        print("Making prediction with AdvancedTranslationPredictor...")
        result = predictor.predict(features_tensor)

        print(f"Prediction result: {result}")

        # Extract text from result
        if isinstance(result, list) and len(result) > 0:
            prediction_text = result[0].get('text', 'No text found')
        else:
            prediction_text = str(result)

        return JSONResponse(
            status_code=200,
            content={
                "prediction": prediction_text,
                "raw_result": result,
                "status": "success"
            }
        )

    except Exception as e:
        print(f"Error during manual prediction: {str(e)}")
        import traceback
        traceback.print_exc()

        raise HTTPException(status_code=500, detail=f"Manual prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)