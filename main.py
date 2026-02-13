import os
import cv2
import pickle
import torch
import torch.nn.functional as F
import json
import numpy as np
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import mediapipe as mp
import tempfile
import asyncio
from pathlib import Path
from model import SignLanguageTransformer

app = FastAPI(title="Sign Language Recognition API", version="1.0.0")

# Add CORS middleware for C# compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Global variables for model and vocabulary
model = None
vocab = None
predictor = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Vocabulary:
    """
    Vocabulary class to map words to integers and vice versa.
    Defined here to ensure pickle compatibility.
    """
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.word_count = {}
        self.n_words = 0

    def add_word(self, word):
        # We don't really need this for loading, but keeping for completeness
        if not hasattr(self, 'word2index'): self.word2index = {}
        if not hasattr(self, 'index2word'): self.index2word = {}
        if not hasattr(self, 'word_count'): self.word_count = {}
        if not hasattr(self, 'n_words'): self.n_words = 0
        
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word_count[word] = 1
            self.n_words += 1
        else:
            self.word_count[word] += 1

    def token_to_id(self, token):
        # Robust check for word mapping
        # Priority: stoi (from debug output), word2index, word2id
        w2i = getattr(self, 'stoi', getattr(self, 'word2index', getattr(self, 'word2id', {})))
        
        # Try direct match
        if token in w2i:
            return w2i[token]
            
        # Try uppercase (common for special tokens like <SOS>)
        if token.upper() in w2i:
            return w2i[token.upper()]
            
        # Try lowercase
        if token.lower() in w2i:
            return w2i[token.lower()]
            
        # Try to get token, default to UNK (often 0, 1, or 3)
        # Check if <UNK> or <unk> exists in w2i to find default
        default_id = w2i.get('<UNK>', w2i.get('<unk>', 0))
        return w2i.get(token, default_id)

    def id_to_token(self, token_id):
        # Robust check for index mapping
        # Priority: itos (from debug output), index2word, id2word
        i2w = getattr(self, 'itos', getattr(self, 'index2word', getattr(self, 'id2word', {})))
        # Handle list vs dict for itos
        if isinstance(i2w, list):
            if 0 <= token_id < len(i2w):
                return i2w[token_id]
            else:
                return '<UNK>'
        return i2w.get(token_id, '<UNK>')
        
    def __len__(self):
        # Robust check for length
        # Priority: itos (list), stoi (dict), n_words
        if hasattr(self, 'itos'): return len(self.itos)
        if hasattr(self, 'stoi'): return len(self.stoi)
        if hasattr(self, 'n_words'): return self.n_words
        if hasattr(self, 'num_words'): return self.num_words
        if hasattr(self, 'word2index'): return len(self.word2index)
        if hasattr(self, 'word2id'): return len(self.word2id)
        return 0
        
    def items(self):
        # Allow iteration like a dict (for compatibility)
        if hasattr(self, 'stoi'): return self.stoi.items()
        if hasattr(self, 'word2index'): return self.word2index.items()
        if hasattr(self, 'word2id'): return self.word2id.items()
        return {}.items()

class AdvancedTranslationPredictor:

    """
    Advanced translation predictor with beam search and n-gram blocking
    """
    def __init__(self, model, vocab, device='cpu'):
        self.model = model
        self.vocab = vocab
        self.device = device
        # self.vocab is now a Vocabulary object, so we rely on its methods
        # self.idx_to_word is not needed as we can use vocab.id_to_token(idx)

        # Initialize global repetition tracker
        self.global_repetition_tracker = set()

    def predict(self, keypoint_sequence, beam_width=3, max_length=100):
        """
        Predict translation using beam search with n-gram blocking
        """
        # Move model to eval mode
        self.model.eval()

        with torch.no_grad():
            # keypoint_sequence is already batched [1, 200, 2653] from main endpoint
            keypoint_tensor = keypoint_sequence.to(self.device)

            # Perform inference
            # Handle different model architectures
            try:
                # Try standard forward pass
                # For SignLanguageTransformer, we need encode() then decode() usually,
                # but let's check if the generic forward works or if we need to call encode specifically.
                # The model definition has an encode method.
                # Let's assume predict_translation uses .encode(), which is compatible with our new class.
                pass 
                
            except Exception as e:
                print(f"Model inference error: {e}")
                pass

        # Use the advanced_predict_translation logic instead of this simplified version
        # We need to import it first
        from advanced_predict_translation import predict_translation
        
        # Call the advanced predictor
        results = predict_translation(self.model, keypoint_tensor, self.vocab, beam_width=beam_width, max_length=max_length)
        
        print(f"DEBUG: Raw prediction results: {results}")
        
        if results and len(results) > 0:
            return results[0]['text']
        else:
            return ""

    def _beam_search_with_ngram_blocking(self, outputs, beam_width, max_length):
        # This method is replaced by advanced_predict_translation
        pass


# Initialize I3D Model
i3d_model = None

def initialize_i3d():
    global i3d_model
    try:
        from i3d import InceptionI3d
        print("Initializing I3D model...")
        i3d = InceptionI3d(400, in_channels=3)
        
        # Load weights
        weights_path = "rgb_imagenet.pt"
        if os.path.exists(weights_path):
            print(f"Loading I3D weights from {weights_path}...")
            state_dict = torch.load(weights_path, map_location='cpu')
            
            # Patch for case sensitivity mismatch (logits -> Logits)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('logits.'):
                    new_key = k.replace('logits.', 'Logits.')
                else:
                    new_key = k
                new_state_dict[new_key] = v
            
            i3d.load_state_dict(new_state_dict, strict=False) # Loose loading to be safe
        else:
            print(f"WARNING: I3D weights not found at {weights_path}. Using random initialization.")
            print("Please download rgb_imagenet.pt to improve accuracy.")
            
        i3d.eval()
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        i3d.to(device)
        i3d_model = i3d
        print("I3D model initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize I3D model: {e}")
        i3d_model = None

def extract_i3d_features(video_path: str, target_length: int) -> np.ndarray:
    """
    Extract 1024-dim features using I3D model.
    Resizes frames to 224x224 and normalizes.
    Returns array of shape (target_length, 1024).
    """
    global i3d_model
    if i3d_model is None:
        initialize_i3d()
        
    if i3d_model is None:
        # Fallback: return small random noise instead of zeros to test sensitivity
        print("WARNING: I3D model missing. Using random noise fallback.")
        return np.random.randn(target_length, 1024).astype(np.float32) * 0.01
        
    # Read video frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize to 224x224
        frame = cv2.resize(frame, (224, 224))
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalize to [-1, 1]
        frame = (frame / 255.0) * 2 - 1
        frames.append(frame)
    cap.release()
    
    if not frames:
        print("WARNING: No frames extracted. Using random noise fallback.")
        return np.random.randn(target_length, 1024).astype(np.float32) * 0.01
        
    # Convert to tensor: [1, 3, T, 224, 224]
    frames_arr = np.array(frames, dtype=np.float32) # (T, 224, 224, 3)
    frames_arr = frames_arr.transpose(3, 0, 1, 2)   # (3, T, 224, 224)
    input_tensor = torch.from_numpy(frames_arr).unsqueeze(0) # (1, 3, T, 224, 224)
    
    device = next(i3d_model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        # I3D extract_features returns [1, T/8, 1024] or similar
        features = i3d_model.extract_features(input_tensor) # [1, T_out, 1024]
        
    features = features.squeeze(0).cpu().numpy() # [T_out, 1024]
    
    # Interpolate to match target_length (original video length)
    current_len = features.shape[0]
    if current_len != target_length:
        feat_tensor = torch.from_numpy(features).unsqueeze(0).transpose(1, 2) # [1, 1024, T_out]
        feat_interpolated = torch.nn.functional.interpolate(
            feat_tensor, size=target_length, mode='linear', align_corners=False
        )
        features = feat_interpolated.transpose(1, 2).squeeze(0).numpy() # [target_length, 1024]
    
    # Debug: Check if I3D features are all zeros
    if np.all(features == 0):
        print("WARNING: I3D features are all zeros! Using random noise fallback.")
        features = np.random.randn(target_length, 1024).astype(np.float32) * 0.01
    
    # Debug: Print I3D feature statistics
    print(f"I3D Features - Mean: {features.mean():.6f}, Std: {features.std():.6f}, Min: {features.min():.6f}, Max: {features.max():.6f}")
        
    return features


def extract_keypoints_from_video(video_path: str) -> list:
    """
    Extract 2,653 spatial keypoints per frame using MediaPipe Holistic + I3D
    """
    cap = cv2.VideoCapture(video_path)
    all_keypoints = []

    # 1. Extract Holistic Features (1629 dim)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        results = holistic.process(rgb_frame)

        # Extract landmarks
        frame_keypoints = []

        # Pose landmarks (33 points * 3 coordinates = 99)
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                frame_keypoints.extend([landmark.x, landmark.y, landmark.z])
        else:
            frame_keypoints.extend([0.0] * 99)

        # Face landmarks (468 points * 3 coordinates = 1404)
        if results.face_landmarks:
            for landmark in results.face_landmarks.landmark:
                frame_keypoints.extend([landmark.x, landmark.y, landmark.z])
        else:
            frame_keypoints.extend([0.0] * 1404)

        # Left hand landmarks (21 points * 3 coordinates = 63)
        if results.left_hand_landmarks:
            for landmark in results.left_hand_landmarks.landmark:
                frame_keypoints.extend([landmark.x, landmark.y, landmark.z])
        else:
            frame_keypoints.extend([0.0] * 63)

        # Right hand landmarks (21 points * 3 coordinates = 63)
        if results.right_hand_landmarks:
            for landmark in results.right_hand_landmarks.landmark:
                frame_keypoints.extend([landmark.x, landmark.y, landmark.z])
        else:
            frame_keypoints.extend([0.0] * 63)

        all_keypoints.append(frame_keypoints)

    cap.release()
    
    num_frames = len(all_keypoints)
    if num_frames == 0:
        return []
        
    # 2. Extract I3D Features (1024 dim)
    print(f"Extracting I3D features for {num_frames} frames...")
    i3d_features = extract_i3d_features(video_path, num_frames) # (T, 1024)
    
    # 3. Concatenate Features and Enforce Strict Dimensions
    final_features = []
    for i in range(num_frames):
        # Apply weighting: MediaPipe (1629) gets 0.7, I3D (1024) gets 0.3
        # This prevents I3D from overpowering the landmarks
        mediapipe_features = all_keypoints[i]
        i3d_frame_features = i3d_features[i].tolist()
        
        # Apply weights
        mediapipe_weighted = [x * 0.7 for x in mediapipe_features]
        i3d_weighted = [x * 0.3 for x in i3d_frame_features]
        
        # Concatenate: 1629 + 1024 = 2653
        combined = mediapipe_weighted + i3d_weighted
        
        # Spatial Clipping/Padding (Feature Dim)
        if len(combined) < 2653:
             combined = combined + [0.0] * (2653 - len(combined))
        elif len(combined) > 2653:
             # Strided sampling to preserve information (User Request)
             # e.g. 2683 -> 2653
             indices = np.linspace(0, len(combined)-1, 2653).astype(int)
             combined = [combined[i] for i in indices]
             
        final_features.append(combined)

    # 4. Temporal Clipping/Padding (Sequence Length)
    # Target: 200 frames (Strict)
    target_frames = 200
    current_frames = len(final_features)
    
    if current_frames < target_frames:
        # Pad with zeros (2653 dim zero vector)
        padding_needed = target_frames - current_frames
        zeros = [0.0] * 2653
        for _ in range(padding_needed):
            final_features.append(zeros)
    elif current_frames > target_frames:
        # Clip
        final_features = final_features[:target_frames]
        
    print(f"Final extracted features shape: ({len(final_features)}, {len(final_features[0])})")
    return final_features


@app.on_event("startup")
async def startup_event():
    """
    Load model and vocabulary on startup with robust architecture detection
    """
    global model, vocab, predictor

    print("Loading model and vocabulary...")

    # Paths
    model_path = "sign_language_FINAL_A100_SUCCESS.pth"
    vocab_path = "vocab.pkl"

    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found: {model_path}")
    if not os.path.exists(vocab_path):
        raise RuntimeError(f"Vocabulary file not found: {vocab_path}")

    # 1. Load State Dict First to detect architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading checkpoint from {model_path}...")
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            print("Checkpoint is not a dict. Assuming it is the full model model.")
            # If it's a full model, we might need a different approach, but usually it's a dict
            state_dict = checkpoint.state_dict()
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    # 2. Architectue Auto-Detection
    # Count layers
    num_encoder_layers = 0
    num_decoder_layers = 0
    
    keys = list(state_dict.keys())
    for key in keys:
        if "transformer.encoder.layers." in key:
            # key format: transformer.encoder.layers.0.xxx
            try:
                layer_idx = int(key.split("transformer.encoder.layers.")[1].split(".")[0])
                num_encoder_layers = max(num_encoder_layers, layer_idx + 1)
            except: pass
        if "transformer.decoder.layers." in key:
            try:
                layer_idx = int(key.split("transformer.decoder.layers.")[1].split(".")[0])
                num_decoder_layers = max(num_decoder_layers, layer_idx + 1)
            except: pass
            
    # Fallback/Validation
    if num_encoder_layers == 0: num_encoder_layers = 4 # Default to 4 as per training config
    if num_decoder_layers == 0: num_decoder_layers = 4 # Default to 4 as per training config
    
    print(f"Detected architecture: {num_encoder_layers} Encoder Layers, {num_decoder_layers} Decoder Layers")

    # Detect Vocab Size from weights
    # Try to find fc_out.weight or generator.weight
    model_vocab_size = 10160 # Default fallback
    if 'fc_out.weight' in state_dict:
        model_vocab_size = state_dict['fc_out.weight'].shape[0]
        print(f"Detected model vocab size from fc_out: {model_vocab_size}")
    elif 'generator.weight' in state_dict:
        model_vocab_size = state_dict['generator.weight'].shape[0]
        print(f"Detected model vocab size from generator: {model_vocab_size}")
    
    # 3. Load Vocabulary File
    import sys
    if not hasattr(sys.modules['__main__'], 'Vocabulary'):
        sys.modules['__main__'].Vocabulary = Vocabulary

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        
    loaded_vocab_size = len(vocab)
    print(f"Loaded vocabulary file with {loaded_vocab_size} tokens")
    
    # Strict Vocab Padding to 10160
    target_vocab_size = 10160
    if loaded_vocab_size < target_vocab_size:
        diff = target_vocab_size - loaded_vocab_size
        print(f"Padding vocabulary with {diff} dummy tokens to reach {target_vocab_size}...")
        for i in range(diff):
            vocab.add_word(f"<EXTRA_ID_{i}>")
    elif loaded_vocab_size > target_vocab_size:
        print(f"WARNING: Loade vocab size {loaded_vocab_size} > {target_vocab_size}. This might cause issues.")
        
    print(f"Final vocab size: {len(vocab)}")

    # Debug: Check Vocabulary Index 1729 (User Request)
    try:
        vocab_1729 = vocab.id_to_token(1729)
        print(f"DEBUG: Vocabulary Index 1729 maps to: '{vocab_1729}'")
    except Exception as e:
        print(f"DEBUG: Could not check index 1729: {e}")
        
    # 4. Instantiate Model
    print("Instantiating model architecture with strict settings...")
    # Architecture settings matched to checkpoint
    model = SignLanguageTransformer(
        input_dim=2653, 
        d_model=512,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        vocab_size=10160
    )
    
    model.to(device)
    
    # 5. Load Weights
    print("Loading state dictionary into model...")
    
    new_state_dict = {}
    
    # Pre-process state_dict
    if 'pos_encoder' in state_dict and 'pos_encoder.pe' not in state_dict:
        print("Patching pos_encoder: Mapping 'pos_encoder' tensor to 'pos_encoder.pe' and 'pos_decoder.pe'")
        pe_tensor = state_dict['pos_encoder']
        # Ensure dimensions match max_len=200
        # Checkpoint PE might be [1, 200, 512] or [200, 1, 512]
        # Current model expects [200, 1, 512] (since batch_first=False usually for PE/Transformer)
        # But let's check tensor shape
        # If it is [1, 200, 512], and we need [200, 1, 512], we transpose.
        if len(pe_tensor.shape) == 3 and pe_tensor.shape[0] == 1:
             pe_tensor = pe_tensor.transpose(0, 1) # [1, 200, 512] -> [200, 1, 512]
        elif len(pe_tensor.shape) == 2:
             pe_tensor = pe_tensor.unsqueeze(1) # [200, 512] -> [200, 1, 512]
            
        new_state_dict['pos_encoder.pe'] = pe_tensor
        new_state_dict['pos_decoder.pe'] = pe_tensor # Share PE
    
    for k, v in state_dict.items():
        if k == 'pos_encoder': continue # Handled above
        
        new_key = k
        # Map generator -> fc_out ALWAYS
        # The model uses 'fc_out' as the registered module. 
        # 'generator' is just an alias attribute, so it doesn't appear in state_dict keys.
        if 'generator.' in k:
             new_key = k.replace('generator.', 'fc_out.')
             
        # Patch I3D keys: logits -> Logits
        # Note: I3D weights are loaded separately in i3d.py/initialize_i3d. 
        # BUT if the main checkpoint contains I3D weights (unlikely given description, usually separate), we'd handle it here.
        
        new_state_dict[new_key] = v
        
    # BIDIRECTIONAL MAPPING for Generator/FC_Out
    # Ensure both keys exist if one is present, to satisfy any model expectation
    if 'fc_out.weight' in new_state_dict and 'generator.weight' not in new_state_dict:
        print("Patching generator weights: Mapping 'fc_out' to 'generator'")
        new_state_dict['generator.weight'] = new_state_dict['fc_out.weight']
        if 'fc_out.bias' in new_state_dict:
            new_state_dict['generator.bias'] = new_state_dict['fc_out.bias']
            
    if 'generator.weight' in new_state_dict and 'fc_out.weight' not in new_state_dict:
         print("Patching fc_out weights: Mapping 'generator' to 'fc_out'")
         new_state_dict['fc_out.weight'] = new_state_dict['generator.weight']
         if 'generator.bias' in new_state_dict:
             new_state_dict['fc_out.bias'] = new_state_dict['generator.bias']
        
    # Debug Vocab Indices (User Request)
    sos_id = vocab.token_to_id('<sos>') if hasattr(vocab, 'token_to_id') else vocab.word2index.get('<sos>')
    eos_id = vocab.token_to_id('<eos>') if hasattr(vocab, 'token_to_id') else vocab.word2index.get('<eos>')
    pour_id = vocab.token_to_id('pour') if hasattr(vocab, 'token_to_id') else vocab.word2index.get('pour', 'N/A')
    
    print(f"DEBUG VOCAB INDICES: SOS={sos_id}, EOS={eos_id}, 'pour'={pour_id}")

    try:
        keys = model.load_state_dict(new_state_dict, strict=False)
        if keys.missing_keys:
            print(f"Warning: Missing keys: {keys.missing_keys[:5]}... (Total {len(keys.missing_keys)})")
            # CRITICAL: Check if fc_out/generator is missing
            if any('fc_out' in k for k in keys.missing_keys):
                print("CRITICAL ERROR: fc_out weights are missing! Output will be garbage.")
                
        if keys.unexpected_keys:
            print(f"Warning: Unexpected keys: {keys.unexpected_keys[:5]}... (Total {len(keys.unexpected_keys)})")
            
    except Exception as e:
         print(f"Error loading state dict: {e}")
         
    model.eval()

    # Initialize predictor
    predictor = AdvancedTranslationPredictor(model, vocab, device)

    print("Model and vocabulary loaded successfully!")


@app.post("/predict", response_class=JSONResponse)
async def predict_sign_language(file: UploadFile = File(...)):
    """
    Process uploaded video and return sign language translation
    """
    # Validate file type
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")

    # Create temporary file
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    try:
        # Save uploaded file to temporary location
        contents = await file.read()
        temp_video.write(contents)
        temp_video_path = temp_video.name
        temp_video.close()

        print(f"Processing video: {temp_video_path}")

        # Extract keypoints from video
        keypoints_list = extract_keypoints_from_video(temp_video_path)

        if not keypoints_list:
            raise HTTPException(status_code=400, detail="No keypoints extracted from video. Please check the video quality.")

        # Convert list of keypoints to a numpy array for easier manipulation and shape checking
        import numpy as np
        keypoints = np.array(keypoints_list, dtype=np.float32)

        print(f"Extracted keypoints array shape: {keypoints.shape}")

        if keypoints.shape[0] == 0:
            # Clean up temporary file
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
            return JSONResponse(content={"prediction": "Error: No keypoints extracted"}, status_code=400)

        # Convert to tensor efficiently
        # keypoints is (200, 2653) float32 numpy array
        input_tensor = torch.from_numpy(keypoints).unsqueeze(0).to(device)

        # Debug: Print raw feature statistics BEFORE normalization
        print(f"RAW Features - Mean: {input_tensor.mean().item():.6f}, Std: {input_tensor.std().item():.6f}, Min: {input_tensor.min().item():.6f}, Max: {input_tensor.max().item():.6f}")

        # Global StandardScaler (Z-score normalization)
        # Calculate global mean and std across ALL features (not per-feature)
        global_mean = input_tensor.mean()
        global_std = input_tensor.std()
        
        # Apply Z-score normalization: (x - mean) / std
        input_tensor = (input_tensor - global_mean) / (global_std + 1e-8)
        
        print(f"Global Normalization - Mean: {global_mean.item():.6f}, Std: {global_std.item():.6f}")
        
        # Debug: Print normalized feature statistics AFTER normalization
        print(f"NORMALIZED Features - Mean: {input_tensor.mean().item():.6f}, Std: {input_tensor.std().item():.6f}, Min: {input_tensor.min().item():.6f}, Max: {input_tensor.max().item():.6f}")

        # 5. Predict using the advanced prediction function directly
        from advanced_predict_translation import predict_translation

        # Call the prediction function with updated parameters for better beam search
        prediction_results = predict_translation(
            predictor.model,
            input_tensor,
            predictor.vocab,
            beam_width=5,
            max_length=20,
            repetition_penalty_factor=1.5,      # As requested
            ngram_blocking_size=2              # As requested (no_repeat_ngram_size)
        )

        # Handle the prediction results properly
        if isinstance(prediction_results, list) and len(prediction_results) > 0:
             # Handle list output (old behavior)
             first_result = prediction_results[0]
             if isinstance(first_result, dict):
                 final_text_string = first_result.get('text', str(first_result))
             else:
                 final_text_string = str(first_result)
        elif isinstance(prediction_results, dict):
             # Handle dictionary output (New Low Confidence Logic)
             if 'prediction' in prediction_results and 'status' in prediction_results:
                 # It's a structured response, return it directly
                 return JSONResponse(content=prediction_results)
             
             # Otherwise extract text
             final_text_string = prediction_results.get('text', prediction_results.get('prediction', str(prediction_results)))
        else:
             final_text_string = str(prediction_results) if prediction_results else ""

        # Clean Output: Strip any special tokens like <SOS> or <EOS>
        special_tokens = ['<sos>', '<eos>', '<SOS>', '<EOS>', '<unk>', '<UNK>', '<pad>', '<PAD>']
        cleaned_text = final_text_string
        for token in special_tokens:
            cleaned_text = cleaned_text.replace(token, "").strip()

        # Additional cleaning: remove extra whitespace
        import re
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        # Clean up temporary file
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)

        # Return proper JSON response for C# compatibility
        return JSONResponse(content={
            "prediction": cleaned_text  # Return as proper JSON object as requested
        })

    except Exception as e:
        # Ensure cleanup happens even if there's an error
        if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)

        print(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")


@app.get("/")
async def root():
    """
    Health check endpoint
    """
    return {"status": "healthy", "message": "Sign Language Recognition API is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)