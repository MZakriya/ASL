# Updated Main.py Documentation

This document describes the updates made to main.py to integrate the new optimized model weights and implement the critical calibration settings.

## Key Changes Made

### 1. Model Loading Updates
- **Model File**: Changed from `v18_ULTIMATE_POLISHED_E5.pth` to `final_sign_language_model.pth`
- **Architecture**: Updated to handle 1536-dim input (512 MediaPipe placeholder + 1024 I3D)
- **Weight Loading**: Implemented new checkpoint structure handling for `model_state_dict`, `src_proj_state_dict`, `tgt_emb_state_dict`, and `fc_out_state_dict`

### 2. Feature Processing Pipeline
- **Padding Function**: Added `pad_to_1536_features()` to pad 1024-dim I3D features with 512-dim zero-padding to match required dimensions (512 zeros + 1024 I3D = 1536)
- **Normalization Function**: Added `normalize_features()` to perform Z-score normalization (mean/std) with epsilon safety
- **Feature Alignment**: Ensures model receives 1536-dim input as specified (model's src_proj layer modified to accept 1536 inputs)

### 3. Advanced Decoding Implementation
- **Nucleus Sampling**: Added `nucleus_sampling()` function with Top-p parameter (p=0.85) and temperature (1.2)
- **Enhanced Predictor**: Updated `AdvancedTranslationPredictor` class with nucleus sampling capability
- **Logit Masking**: Implemented masking of dead indices (19122, 18275) to -1e9 during every decoding step
- **Replaced greedy search** with nucleus sampling as the primary decoding strategy

### 4. Vocabulary Processing
- **Cleaning Function**: Added `clean_vocab_word()` to remove metadata suffixes (like 'en6', 'en7') and special characters
- **Integration**: Applied cleaning in the `_decode_sequence()` method to produce cleaner output

### 5. New Endpoints
- **Updated `/predict`**: Now handles 1024-dim I3D features, pads to 1536-dim, normalizes, and uses nucleus sampling
- **Added `/test-local`**: Validates model performance against `laptop_test_features.npy` with comparison to actual captions

### 6. Configuration Updates
- **Strict Calibrated Nucleus Sampling**: p=0.3, temperature=0.4 for high confidence word selection
- **Hard-coded Parameters**: Values enforced in decoding function to ensure consistency
- **Dynamic Repetition Penalty**: -50.0 penalty to logits of already-generated words to prevent loops
- **Vocabulary Common Sense Filter**: Penalizes words with index > 5000, focuses on common words
- **Aggressive Motion Amplification**: Normalized features multiplied by 3.0 for Std 2.0-3.0 range
- **Length Guidance**: Encourages 4-6 word sentences with EOS triggering for concise output

## Technical Specifications

### Input Processing Pipeline:
1. Extract 1024-dim I3D features (from video or file)
2. Pad with 512-dim zeros to create 1536-dim input
3. Apply Z-score normalization (mean=0, std=1 with 1e-6 epsilon)
4. Feed to Transformer model

### Decoding Strategy:
1. Nucleus (Top-p) Sampling with p=0.85
2. Temperature scaling at 1.2 for diversity
3. Logit masking of problematic indices (19122, 18275)
4. Maximum sequence length of 50 tokens
5. Vocabulary cleaning for final output

## Usage

### Standard Video Prediction:
POST to `/predict` with video file

### Local Validation:
POST to `/test-local` to run validation on `laptop_test_features.npy`

## Error Handling

- Robust handling of missing or malformed test data
- Graceful degradation with informative error messages
- Unicode compatibility fixes for Windows environments
- Comprehensive logging for debugging

## Architecture Compatibility

- Maintains FastAPI structure and middleware
- Preserves existing CORS settings
- Compatible with MediaPipe feature extraction
- Handles different numpy file formats and versions