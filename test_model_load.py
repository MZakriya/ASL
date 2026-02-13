
import torch
import torch.nn as nn
import pickle
import json
import os
from model import SignLanguageTransformer

def test_load():
    print("Testing model loading...")
    
    # 1. Load Vocab for size
    vocab_path = "vocab.pkl"
    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        vocab_size = len(vocab)
        print(f"Vocab size: {vocab_size}")
    else:
        print("Vocab not found, using default 1000")
        vocab_size = 1000

    # 2. Load Config
    config_path = "config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        model_params = config.get('model_params', {})
        print(f"Config params: {model_params}")
    else:
        model_params = {}

    # 3. Instantiate Model
    # Map config keys to model args
    # config: input_dim, model_dim, num_heads, num_layers
    # model: input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers...
    
    model = SignLanguageTransformer(
        input_dim=model_params.get('input_dim', 2653),
        d_model=model_params.get('model_dim', 512),
        nhead=model_params.get('num_heads', 8),
        num_encoder_layers=model_params.get('num_layers', 6),
        num_decoder_layers=model_params.get('num_layers', 6),
        vocab_size=vocab_size
    )
    
    # 4. Load Weights
    model_path = "sign_language_FINAL_A100_SUCCESS.pth"
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return

    try:
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Check if it's a full model or state dict
        if not isinstance(state_dict, dict):
            print("Loaded object is not a dict. It might be a full model.")
            # If it works, great.
            return

        print("Object is a dictionary. Attempting load_state_dict...")
        
        # strict=False to allow soft matching, but we want to see errors first
        keys_loaded = model.load_state_dict(state_dict, strict=False)
        
        print("\n--- Load Results ---")
        if len(keys_loaded.missing_keys) > 0:
            print(f"Missing keys ({len(keys_loaded.missing_keys)}):")
            for k in keys_loaded.missing_keys[:10]:
                print(f"  {k}")
            if len(keys_loaded.missing_keys) > 10: print("  ...")
            
        if len(keys_loaded.unexpected_keys) > 0:
            print(f"Unexpected keys ({len(keys_loaded.unexpected_keys)}):")
            for k in keys_loaded.unexpected_keys[:10]:
                print(f"  {k}")
            if len(keys_loaded.unexpected_keys) > 10: print("  ...")
            
        if len(keys_loaded.missing_keys) == 0 and len(keys_loaded.unexpected_keys) == 0:
            print("✅ Model loaded perfectly!")
        else:
            print("❌ Model mismatch!")

    except Exception as e:
        print(f"Error during loading: {e}")

if __name__ == "__main__":
    test_load()
