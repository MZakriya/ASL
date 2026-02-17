import sys
sys.path.append("d:\\All Projects\\ASLR")
from main import Vocabulary

sys.stdout = open("tests/vocab_dump.txt", "w", encoding='utf-8')

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
    
print(f"Vocab Type: {type(vocab)}")
print(f"Expected UNK ID: {getattr(vocab, 'unk_index', 'Not Found')}")

# Check ITOS
if hasattr(vocab, 'itos'):
    itos = vocab.itos
    print(f"ITOS Length: {len(itos)}")
    print(f"ID 0: {itos[0]}")
    print(f"ID 1: {itos[1]}")
    print(f"ID 2: {itos[2]}")
    print(f"ID 3: {itos[3]}")
    print(f"ID 4: {itos[4]}")
    
    # Search for "i", "will", "see"
    stoi = vocab.stoi
    print(f"\nTarget Words:")
    print(f"'i': {stoi.get('i', 'Not Found')}")
    print(f"'will': {stoi.get('will', 'Not Found')}")
    print(f"'see': {stoi.get('see', 'Not Found')}")
    print(f"'you': {stoi.get('you', 'Not Found')}")
    print(f"'again': {stoi.get('again', 'Not Found')}")

print("\nDone.")
