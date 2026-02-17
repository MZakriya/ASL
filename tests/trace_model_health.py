import sys
import os
import torch
import pickle

# Redirect stdout to root file
sys.stdout = open("trace_health.log", "w")
sys.stderr = sys.stdout

print("Starting Model Health Check...")

try:
    sys.path.append("d:\\All Projects\\ASLR")
    from model import SignLanguageTransformer
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = SignLanguageTransformer(vocab_size=9967, d_model=512)
    checkpoint = torch.load("v18_ULTIMATE_POLISHED_E5.pth", map_location=device)
    
    # Load Main Layers
    if 'src_proj' in checkpoint: model.src_proj.load_state_dict(checkpoint['src_proj'])
    if 'fc_out' in checkpoint: model.fc_out.load_state_dict(checkpoint['fc_out'])
    if 'tgt_emb' in checkpoint: model.tgt_emb.load_state_dict(checkpoint['tgt_emb'])
    
    model.to(device)
    model.eval()
    
    # Check Weights
    print("\n--- Weight Stats ---")
    src_w = model.src_proj.weight
    fc_w = model.fc_out.weight
    
    print(f"Src Proj Mean: {src_w.mean().item():.6f}, Std: {src_w.std().item():.6f}, Max: {src_w.abs().max().item():.6f}")
    if src_w.abs().mean() < 1e-5:
        print("WARNING: Src Proj weights are near ZERO! Implementation invalid.")
        
    print(f"FC Out Mean: {fc_w.mean().item():.6f}, Std: {fc_w.std().item():.6f}")
    
    # Check "this" token bias
    # Assuming "this" is around index 15 (from prev logs)
    # Actually let's just look at max logits for EOS input
    
    print("\n--- Logit Bias Check ---")
    # Decoder Input: SOS
    ys = torch.ones(1, 1).long().to(device)
    # Memory: Zeros (to see pure bias)
    memory = torch.zeros(1, 200, 512).to(device)
    
    with torch.no_grad():
        out = model.decode(ys, memory) 
        logits = model.fc_out(out[:, -1])
        
    print(f"Logits (Zero Context) Max: {logits.max().item():.4f}")
    probs = torch.softmax(logits, dim=-1)
    topv, topi = torch.topk(probs, 5)
    
    for i in range(5):
        print(f"Top {i}: ID={topi[0][i].item()} Prob={topv[0][i].item():.4f}")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()

print("End of Health Check.")
