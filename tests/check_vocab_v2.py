import sys
import pickle
import os

# Robust unbuffered stdout
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1, encoding='utf-8', closefd=False)

print("Starting Vocab Check...")

try:
    sys.path.append("d:\\All Projects\\ASLR")
    from main import Vocabulary
    print("Imported Vocabulary class.")
    
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    print(f"Loaded vocab. Type: {type(vocab)}")
    
    if hasattr(vocab, 'unk_index'):
        print(f"unk_index attribute: {vocab.unk_index}")
    else:
        print("No unk_index attribute found.")
        
    if hasattr(vocab, 'itos'):
        print(f"ITOS Length: {len(vocab.itos)}")
        print(f"ID 0: '{vocab.itos[0]}'")
        print(f"ID 1: '{vocab.itos[1]}'")
        print(f"ID 2: '{vocab.itos[2]}'")
        print(f"ID 3: '{vocab.itos[3]}'")
        
        # Search for words
        targets = ["i", "will", "see", "you", "again", "this"]
        stoi = getattr(vocab, 'stoi', {})
        for w in targets:
            print(f"Word '{w}': {stoi.get(w, 'Not Found')}")
            
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("Done.")
