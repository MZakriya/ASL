import torch
import os
import sys

# Path to the model file
model_path = r"d:\All Projects\ASLR\sign_language_A100_v10_FINAL_FIX.pth"

def audit_model():
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    try:
        print(f"Loading model from {model_path}...")
        # Load on CPU to avoid CUDA requirements for this simple check
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        state_dict = checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("Found 'model_state_dict' key.")
        
        # Check for generator or fc_out weights
        keys_to_check = ['module.generator.weight', 'generator.weight', 'module.fc_out.weight', 'fc_out.weight']
        
        found = False
        for key in keys_to_check:
            if key in state_dict:
                weight = state_dict[key]
                print(f"\n[FOUND] Layer '{key}'")
                print(f"  Shape: {weight.shape}")
                print(f"  Output Dimension (Vocab Size): {weight.shape[0]}")
                print(f"  Input Dimension: {weight.shape[1]}")
                found = True
                break
        
        if not found:
            print("\n[WARNING] Could not find 'generator.weight' or 'fc_out.weight' in state_dict.")
            print("Available keys (first 10):")
            for i, k in enumerate(list(state_dict.keys())[:10]):
                print(f"  {k}")

    except Exception as e:
        print(f"Error auditing model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    audit_model()
