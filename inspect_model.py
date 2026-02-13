
import torch
import os

model_path = "sign_language_FINAL_A100_SUCCESS.pth"
if not os.path.exists(model_path):
    print(f"Model file not found: {model_path}")
    exit(1)

try:
    print(f"Loading {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f"Type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print("Keys found in checkpoint:")
        keys = list(checkpoint.keys())
        # Print first 20 keys to get an idea of structure
        for key in keys[:20]:
            print(f"  - {key}")
            
        # Check for specific architecture hints
        if 'encoder.layers.0.self_attn.in_proj_weight' in keys:
            print("\nLooks like a Transformer Encoder!")
        if 'decoder.layers.0.self_attn.in_proj_weight' in keys:
            print("\nLooks like a Transformer Decoder!")
            
    else:
        print("Checkpoint is not a dictionary. Typically this means it's a full model object.")
        print(checkpoint)

except Exception as e:
    print(f"Error loading model: {e}")
