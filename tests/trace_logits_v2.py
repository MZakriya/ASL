import sys
import os

# Redirect stdout/stderr
sys.stdout = open("tests/trace_out.log", "w")
sys.stderr = sys.stdout

print("Starting trace...")

try:
    import torch
    import pickle
    sys.path.append("d:\\All Projects\\ASLR")
    from model import SignLanguageTransformer
    
    print("Imports done.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    print("Vocab loaded.")
        
    model = SignLanguageTransformer(vocab_size=9967, d_model=512)
    checkpoint = torch.load("v18_ULTIMATE_POLISHED_E5.pth", map_location=device)
    
    if 'fc_out' in checkpoint: model.fc_out.load_state_dict(checkpoint['fc_out'])
    if 'tgt_emb' in checkpoint: model.tgt_emb.load_state_dict(checkpoint['tgt_emb'])
    
    model.to(device)
    model.eval()
    print("Model loaded.")
    
    # Dummy Encoder Output (Batch=1, Len=200, Dim=512)
    memory = torch.randn(1, 200, 512).to(device)
    
    # Decoder Input: SOS
    ys = torch.ones(1, 1).long().to(device) # SOS=1
    
    print("Running decode step...")
    with torch.no_grad():
        # Just simple forward pass of decoder to see bias
        # We need to manually call decode/fc_out like in greedy search
        # model.decode(tgt, memory)
        out = model.decode(ys, memory) # [1, 1, 512]
        logits = model.fc_out(out[:, -1]) # [1, 9967]
        
    print(f"Logits Stats: Min={logits.min().item():.4f}, Max={logits.max().item():.4f}")
    
    probs = torch.softmax(logits, dim=-1)
    topv, topi = torch.topk(probs, 10)
    
    print("\nTop 10 Predictions (SOS context):")
    for i in range(10):
        idx = topi[0][i].item()
        prob = topv[0][i].item()
        logit = logits[0][idx].item()
        
        word = "UNK"
        if hasattr(vocab, 'itos'):
             if isinstance(vocab.itos, list) and idx < len(vocab.itos): word = vocab.itos[idx]
             elif isinstance(vocab.itos, dict): word = vocab.itos.get(idx, "UNK")
             
        print(f"{i+1}. '{word}' (ID: {idx}) | Prob: {prob:.6f} | Logit: {logit:.4f}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("Done.")
