import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append("d:\\All Projects\\ASLR")

from advanced_predict_translation import AdvancedTranslationPredictor
from main import Vocabulary

# Mock Model and Vocab
class MockModel(torch.nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.encode = lambda x: torch.randn(1, 10, d_model)
        self.decode = lambda x, y, z=None: torch.randn(1, x.size(1), d_model)
        self.fc_out = torch.nn.Linear(d_model, vocab_size)
        self.eos_id = 2
        
    def generate_square_subsequent_mask(self, sz):
        return torch.zeros(sz, sz)

def test_inference_robustness():
    print("Running Inference Robustness Test...")
    
    # Setup
    vocab_size = 100
    d_model = 32
    model = MockModel(vocab_size, d_model)
    
    # Mock Vocab with itos list
    vocab = Vocabulary()
    vocab.itos = [f"word_{i}" for i in range(vocab_size)]
    vocab.itos[0] = "<PAD>"
    vocab.itos[1] = "<SOS>"
    vocab.itos[2] = "<EOS>"
    vocab.itos[3] = "<UNK>"
    vocab.stoi = {w: i for i, w in enumerate(vocab.itos)}
    
    # Config
    config = {
        'max_length': 20,
        'beam_width': 5, # Updated to 5
        'strict_repetition_penalty': 50.0, # Updated to 50.0 (Cumulative)
        'temperature': 0.4
    }
    
    predictor = AdvancedTranslationPredictor(model, vocab, config)
    
    # Dummy Input
    encoder_input = torch.randn(1, 50, 2653) 
    
    # Predict
    print("\nTesting Greedy Search with Relaxed Penalty:")
    results = predictor.predict(encoder_input)
    print("Results:", results)
    
    # Check if we get a result
    if not results:
        print("FAIL: No results returned")
        return
        
    seq = results[0]['sequence']
    text = results[0]['text']
    
    print(f"Generated Sequence: {seq}")
    print(f"Generated Text: {text}")
    
    # Check for repetition
    # With random model, we might get random repeats, but penalty should suppress them
    # We can't strictly potential correctness with random model, but we check for CRASHES or empty strings
    
    if len(seq) < 2:
        print("WARNING: Sequence too short (likely just SOS/EOS)")
        
    print("\nTest Complete.")

if __name__ == "__main__":
    test_inference_robustness()
