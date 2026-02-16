import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import pickle
import numpy as np
import argparse

# Attempt to import key components from main.py or define them
# Assuming main.py is in the current directory
sys.path.append(os.getcwd())

def get_model_and_vocab():
    # Load Vocabulary
    vocab_path = "vocab.pkl"
    if not os.path.exists(vocab_path):
        print(f"Error: {vocab_path} not found.")
        return None, None
        
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        
    # Check if Vocabulary class is available
    if not hasattr(sys.modules['__main__'], 'Vocabulary'):
         # We might need to define it if pickle fails to find class
         class Vocabulary:
            def __init__(self, freq_threshold):
                self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
                self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
                self.freq_threshold = freq_threshold
         sys.modules['__main__'].Vocabulary = Vocabulary

    # Initialize Model
    from model import SignLanguageTransformer
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SignLanguageTransformer(
        input_dim=2653, 
        d_model=512,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        vocab_size=10160
    ).to(device)
    
    # Load Weights
    checkpoint_path = "sign_language_A100_v10_FINAL_FIX.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # Patching logic (simplified from main.py)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('generator.', 'fc_out.')
            new_state_dict[new_key] = v
            
        # PE Patch
        if 'pos_encoder' in state_dict:
             pe = state_dict['pos_encoder']
             if len(pe.shape) == 3 and pe.shape[0] == 1: pe = pe.transpose(0, 1)
             new_state_dict['pos_encoder.pe'] = pe
             new_state_dict['pos_decoder.pe'] = pe
             
        model.load_state_dict(new_state_dict, strict=False)
    else:
        print("Warning: Checkpoint not found, starting fresh (unlikely to work well).")

    return model, vocab

def train_single_sample(video_path, target_text, epochs=50):
    from main import extract_keypoints_from_video
    
    model, vocab = get_model_and_vocab()
    if not model: return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi['<PAD>'])
    
    # Process Data
    print(f"Extracting features from {video_path}...")
    # This calls the main.py function which handles extraction + interpolation
    features = extract_keypoints_from_video(video_path)
    
    # Verify shape
    features = np.array(features)
    if features.shape[1] > 2653: features = features[:, :2653] # Strict clip
    
    input_tensor = torch.from_numpy(features).unsqueeze(0).to(device) # (1, 200, 2653)
    
    # Process Target
    tokens = [vocab.stoi['<SOS>']]
    for word in target_text.lower().split():
        if word in vocab.stoi:
            tokens.append(vocab.stoi[word])
        else:
            tokens.append(vocab.stoi['<UNK>'])
    tokens.append(vocab.stoi['<EOS>'])
    
    target_tensor = torch.tensor(tokens).unsqueeze(0).to(device) # (1, L)
    
    print(f"Target Sequence: {tokens}")
    print(f"Starting Overfit Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward
        # Decoder Input: Target excluding last token (<EOS>)
        # Target Output: Target excluding first token (<SOS>)
        tgt_input = target_tensor[:, :-1]
        tgt_output = target_tensor[:, 1:]
        
        output = model(input_tensor, tgt_input) # (1, L-1, Vocab)
        
        # Reshape for Loss
        output = output.reshape(-1, output.shape[-1])
        tgt_output = tgt_output.reshape(-1)
        
        loss = criterion(output, tgt_output)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.4f}")
            
    print("Training Complete. Saving overfit_model.pth...")
    torch.save(model.state_dict(), "overfit_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--text", type=str, required=True, help="Ground truth text")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    args = parser.parse_args()
    
    train_single_sample(args.video, args.text, args.epochs)
