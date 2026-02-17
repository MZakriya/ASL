import torch
import sys
import os
import pickle
sys.path.append("d:\\All Projects\\ASLR")
from model import SignLanguageTransformer

def trace_logits():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading resources...")
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
        
    model = SignLanguageTransformer(vocab_size=9967, d_model=512)
    checkpoint = torch.load("v18_ULTIMATE_POLISHED_E5.pth", map_location=device)
    
    if 'fc_out' in checkpoint: model.fc_out.load_state_dict(checkpoint['fc_out'])
    if 'tgt_emb' in checkpoint: model.tgt_emb.load_state_dict(checkpoint['tgt_emb'])
    # Load others just in case, but we test decoder mainly
    model.to(device)
    model.eval()
    
    # Dummy Encoder Output (Batch=1, Len=200, Dim=512)
    memory = torch.randn(1, 200, 512).to(device)
    
    # Decoder Input: SOS
    ys = torch.ones(1, 1).long().to(device) # SOS=1
    
    print("Running decode step...")
    with torch.no_grad():
        out = model.decode(ys, memory) # [1, 1, 512]
        logits = model.fc_out(out[:, -1]) # [1, 9967]
        
    print(f"Logits Shape: {logits.shape}")
    print(f"Logits Stats: Min={logits.min().item():.4f}, Max={logits.max().item():.4f}, Mean={logits.mean().item():.4f}")
    
    # Top 10
    probs = torch.softmax(logits, dim=-1)
    topv, topi = torch.topk(probs, 10)
    
    output_lines = []
    output_lines.append(f"Logits Statistics:")
    output_lines.append(f"Min: {logits.min().item()}")
    output_lines.append(f"Max: {logits.max().item()}")
    
    output_lines.append("\nTop 10 Predictions:")
    for i in range(10):
        idx = topi[0][i].item()
        prob = topv[0][i].item()
        logit = logits[0][idx].item()
        word = vocab.itos[idx] if hasattr(vocab, 'itos') else str(idx)
        output_lines.append(f"{i+1}. '{word}' (ID: {idx}) | Prob: {prob:.6f} | Logit: {logit:.4f}")
        
    with open("tests/logits_dump.txt", "w") as f:
        f.write("\n".join(output_lines))
        
    print("Done. Wrote to tests/logits_dump.txt")

if __name__ == "__main__":
    trace_logits()
