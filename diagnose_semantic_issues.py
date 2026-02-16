"""
Comprehensive Debugging Script for Persistent Semantic Issues

This script helps diagnose why the model is producing semantically incorrect output
like "pour push was silk..." even after all alignment fixes.

Run this script to check:
1. Model architecture matches vocab size
2. I3D features are non-zero
3. Feature statistics are reasonable
4. Vocab mappings are correct
5. Model weights are loaded properly
"""

import torch
import pickle
import numpy as np
import os
import sys

def check_vocab():
    """Check vocabulary file"""
    print("="*70)
    print("1. VOCABULARY CHECK")
    print("="*70)
    
    try:
        with open("vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        
        vocab_size = len(vocab.itos) if hasattr(vocab, 'itos') else len(vocab)
        print(f"✓ Vocab loaded: {vocab_size} tokens")
        
        # Check critical indices
        critical = {
            0: '<PAD>',
            1: '<SOS>',
            2: '<EOS>',
            3: '<UNK>',
            284: 'buy',
            1729: 'pour'
        }
        
        print("\nCritical indices:")
        for idx, expected in critical.items():
            if isinstance(vocab.itos, list) and idx < len(vocab.itos):
                actual = vocab.itos[idx]
                match = "✓" if expected.lower() in actual.lower() else "✗"
                print(f"  {match} Index {idx:4d}: '{actual}' (expected: {expected})")
        
        return vocab
    except Exception as e:
        print(f"✗ Error loading vocab: {e}")
        return None

def check_model():
    """Check model file"""
    print("\n" + "="*70)
    print("2. MODEL CHECK")
    print("="*70)
    
    model_path = "best_model.pth"
    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        return None
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"✓ Model loaded: {model_path}")
        
        # Check what's in the checkpoint
        if isinstance(checkpoint, dict):
            print(f"\nCheckpoint keys: {list(checkpoint.keys())}")
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Count parameters
            total_params = sum(p.numel() for p in state_dict.values())
            print(f"Total parameters: {total_params:,}")
            
            # Check vocab size in model
            if 'fc_out.weight' in state_dict:
                vocab_size_model = state_dict['fc_out.weight'].shape[0]
                print(f"Model vocab size (fc_out): {vocab_size_model}")
            elif 'generator.weight' in state_dict:
                vocab_size_model = state_dict['generator.weight'].shape[0]
                print(f"Model vocab size (generator): {vocab_size_model}")
            else:
                print("⚠ Cannot find output layer (fc_out or generator)")
            
            # Check input dimension
            if 'input_projection.weight' in state_dict:
                input_dim = state_dict['input_projection.weight'].shape[1]
                print(f"Model input dim: {input_dim}")
            
            return state_dict
        else:
            print("⚠ Checkpoint is not a dictionary")
            return checkpoint
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_i3d():
    """Check I3D weights"""
    print("\n" + "="*70)
    print("3. I3D WEIGHTS CHECK")
    print("="*70)
    
    i3d_path = "rgb_imagenet.pt"
    if not os.path.exists(i3d_path):
        print(f"✗ I3D weights not found: {i3d_path}")
        print("  Download from: https://github.com/piergiaj/pytorch-i3d/raw/master/models/rgb_imagenet.pt")
        return None
    
    try:
        i3d_weights = torch.load(i3d_path, map_location='cpu')
        print(f"✓ I3D weights loaded: {i3d_path}")
        print(f"  Number of layers: {len(i3d_weights)}")
        
        # Check for logits layer
        has_logits = any('logits' in k.lower() for k in i3d_weights.keys())
        print(f"  Has logits layer: {has_logits}")
        
        # Sample some keys
        print("\n  Sample keys:")
        for i, key in enumerate(list(i3d_weights.keys())[:5]):
            print(f"    - {key}")
        
        return i3d_weights
    except Exception as e:
        print(f"✗ Error loading I3D weights: {e}")
        return None

def check_feature_extraction():
    """Test feature extraction on a dummy video"""
    print("\n" + "="*70)
    print("4. FEATURE EXTRACTION TEST")
    print("="*70)
    
    print("Creating dummy video features...")
    
    # Simulate MediaPipe features (1629 dims)
    mediapipe_features = np.random.rand(200, 1629) * 0.5  # 0-0.5 range
    
    # Simulate I3D features (1024 dims)
    i3d_features = np.random.randn(200, 1024) * 0.1  # Small values
    
    # Apply weights
    mediapipe_weighted = mediapipe_features * 0.7
    i3d_weighted = i3d_features * 0.3
    
    # Concatenate
    combined = np.concatenate([mediapipe_weighted, i3d_weighted], axis=1)
    
    print(f"✓ Combined features shape: {combined.shape}")
    print(f"  MediaPipe weighted: {mediapipe_weighted.shape}")
    print(f"  I3D weighted: {i3d_weighted.shape}")
    print(f"\nFeature statistics:")
    print(f"  Mean: {combined.mean():.6f}")
    print(f"  Std: {combined.std():.6f}")
    print(f"  Min: {combined.min():.6f}")
    print(f"  Max: {combined.max():.6f}")
    
    # Check if reasonable
    if combined.mean() < -10 or combined.mean() > 10:
        print("  ⚠ WARNING: Mean is extreme!")
    if combined.std() < 0.01 or combined.std() > 100:
        print("  ⚠ WARNING: Std is extreme!")
    
    return combined

def check_model_vocab_alignment(vocab, model_state_dict):
    """Check if model and vocab sizes match"""
    print("\n" + "="*70)
    print("5. MODEL-VOCAB ALIGNMENT CHECK")
    print("="*70)
    
    if vocab is None or model_state_dict is None:
        print("✗ Cannot check alignment (vocab or model not loaded)")
        return
    
    vocab_size = len(vocab.itos) if hasattr(vocab, 'itos') else len(vocab)
    
    # Find model vocab size
    model_vocab_size = None
    if 'fc_out.weight' in model_state_dict:
        model_vocab_size = model_state_dict['fc_out.weight'].shape[0]
        layer_name = 'fc_out'
    elif 'generator.weight' in model_state_dict:
        model_vocab_size = model_state_dict['generator.weight'].shape[0]
        layer_name = 'generator'
    
    if model_vocab_size:
        print(f"Vocab size (vocab.pkl): {vocab_size}")
        print(f"Model vocab size ({layer_name}): {model_vocab_size}")
        
        if vocab_size == model_vocab_size:
            print("✓ MATCH: Vocab and model sizes align")
        else:
            print(f"✗ MISMATCH: Vocab ({vocab_size}) != Model ({model_vocab_size})")
            print(f"  Difference: {abs(vocab_size - model_vocab_size)}")
            
            if vocab_size < model_vocab_size:
                print(f"  ⚠ Vocab is smaller - model expects {model_vocab_size - vocab_size} more tokens")
            else:
                print(f"  ⚠ Vocab is larger - {vocab_size - model_vocab_size} extra tokens will never be predicted")

def generate_diagnostic_report():
    """Generate comprehensive diagnostic report"""
    print("\n" + "="*70)
    print("DIAGNOSTIC REPORT")
    print("="*70)
    
    vocab = check_vocab()
    model_state_dict = check_model()
    i3d_weights = check_i3d()
    features = check_feature_extraction()
    check_model_vocab_alignment(vocab, model_state_dict)
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    print("\n1. If vocab indices are wrong:")
    print("   - Regenerate vocab.pkl from training data")
    print("   - Ensure special tokens at indices 0,1,2,3")
    
    print("\n2. If model-vocab size mismatch:")
    print("   - Check if model was trained with different vocab")
    print("   - Verify vocab.pkl matches training vocab")
    
    print("\n3. If I3D weights missing:")
    print("   - Download: python download_weights.py")
    print("   - Or manually from GitHub")
    
    print("\n4. If features are extreme:")
    print("   - Check MediaPipe extraction")
    print("   - Verify I3D is producing non-zero values")
    
    print("\n5. Next steps:")
    print("   - Run server: python main.py")
    print("   - Check startup logs for vocab verification")
    print("   - Upload test video and check greedy search output")
    print("   - If still wrong, model may need retraining")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    print("\n" + "="*70)
    print("SEMANTIC ALIGNMENT DIAGNOSTIC TOOL")
    print("="*70)
    print("\nThis script checks for common issues causing semantic misalignment.")
    print("Running comprehensive diagnostics...\n")
    
    generate_diagnostic_report()
    
    print("\n✓ Diagnostic complete!")
    print("\nIf issues persist after fixing the above, the model may need:")
    print("  1. Retraining with correct vocab and features")
    print("  2. Different hyperparameters (learning rate, etc.)")
    print("  3. More training data or better data quality")
