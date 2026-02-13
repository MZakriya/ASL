
import pickle
import sys

# Define a minimal classes to match whatever might be in there
class Vocabulary:
    pass

# Patch main
if not hasattr(sys.modules['__main__'], 'Vocabulary'):
    sys.modules['__main__'].Vocabulary = Vocabulary

try:
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
        
    print(f"Loaded vocab type: {type(vocab)}")
    print(f"Dir: {dir(vocab)}")
    if hasattr(vocab, '__dict__'):
        print(f"Dict keys: {vocab.__dict__.keys()}")
        
except Exception as e:
    print(f"Error: {e}")
