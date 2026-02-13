
import torch
import torch.nn.functional as F
from advanced_predict_translation import AdvancedTranslationPredictor
import logging

# Configure logging to show info
logging.basicConfig(level=logging.INFO)

class MockVocab:
    def __init__(self):
        self.word2id = {
            '<sos>': 0, '<eos>': 1, '<pad>': 2,
            'hello': 3, 'world': 4, 'is': 5, 'the': 6, 'very': 7, 'good': 8
        }
        self.id2word = {v: k for k, v in self.word2id.items()}

    def token_to_id(self, token):
        return self.word2id.get(token, 2)

    def id_to_token(self, token_id):
        return self.id2word.get(token_id, '<unk>')

    def __len__(self):
        return len(self.word2id)

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_size = 9
        self.d_model = 16

    def encode(self, x):
        # Return dummy encoder output
        return torch.randn(1, 10, self.d_model)

    def decode(self, tgt, memory):
        # Return dummy decoder output
        return torch.randn(1, tgt.size(1), self.d_model)
    
    def generator(self, x):
        # Return fixed logits to test penalties
        # Logits favor 'is' (id 5) and 'the' (id 6)
        logits = torch.zeros(self.vocab_size)
        logits[5] = 10.0  # High score for 'is'
        logits[6] = 9.0   # High score for 'the'
        logits[3] = 5.0   # Medium for 'hello'
        logits[1] = 0.0   # Low for EOS
        return logits.unsqueeze(0) # Batch size 1

def run_verification():
    print("Starting verification of refined Decoder logic...")
    
    vocab = MockVocab()
    model = MockModel()
    
    # Configuration with our new defaults
    config = {
        'frequency_penalty_weight': 0.5, # 50% reduction
        'ngram_blocking_size': 3,
        'beam_width': 1 # simplifying for test
    }
    
    predictor = AdvancedTranslationPredictor(model, vocab, config=config)
    
    # 1. Test Frequency Penalty
    print("\n--- Testing Frequency Penalty Scaling ---")
    
    # Manually call _apply_frequency_penalty to check math
    # Create dummy probabilities
    probs = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.1, 0.0, 0.0])
    # bias words: 'is' (5), 'the' (6) are in default list? 
    # Let's check the predictor's bias words
    print(f"Bias words in predictor: {predictor.bias_word_ids}")
    # We need to make sure 'is' and 'the' are in there. They should be by default.
    
    print(f"Original prob of 'is' (id 5): {probs[5].item()}")
    
    constrained_probs = predictor._apply_frequency_penalty(probs.clone())
    
    print(f"Penalized prob of 'is' (id 5): {constrained_probs[5].item()}")
    
    expected_prob = 0.4 * (1.0 - 0.5)
    print(f"Expected prob: {expected_prob}")
    
    if abs(constrained_probs[5].item() - expected_prob) < 1e-5:
        print("✅ Frequency penalty applied correctly (scaling)!")
    else:
        print("❌ Frequency penalty failed!")

    # 2. Test N-Gram Soft Penalty
    print("\n--- Testing N-Gram Soft Penalty ---")
    # Context: "very very" (ids 7, 7) -> next is "very" (7) again?
    # ngram_blocking_size = 3
    # history: [7, 7]
    # We want to see if adding another 7 is penalized
    
    ngram_hist = [7, 7, 7, 7] # "... very very very"
    # The refined logic looks at last N-1 tokens. 
    # blocking size = 3. Context = last 2 tokens: [7, 7]
    # It checks providing if [7, 7] appeared before.
    # In [7, 7, 7, 7], the context [7, 7] appears at indices 0 (tokens 0,1) and 1 (tokens 1,2)
    # The token following the first [7, 7] is 7.
    # So 7 should be penalized.
    
    probs = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5]) # 50% for 7, 50% for 8
    print(f"Original prob of 'very' (id 7): {probs[7].item()}")
    
    constrained_probs = predictor._apply_ngram_penalty(probs.clone(), ngram_hist)
    
    print(f"Penalized prob of 'very' (id 7): {constrained_probs[7].item()}")
    
    # Logic: probs[7] *= 0.1
    expected_prob = 0.5 * 0.1
    # Note: re-normalization happens outside, so we check the raw penalized value before norm
    # The function _apply_ngram_penalty does re-normalize at the end!
    # So we need to calculate expected after normalization.
    
    raw_penalized_7 = 0.5 * 0.1 # 0.05
    raw_8 = 0.5 # 0.5
    total = 0.05 + 0.5 # 0.55
    expected_normalized_7 = 0.05 / 0.55
    
    print(f"Expected penalized prob (approx): {expected_normalized_7:.4f}")
    
    if abs(constrained_probs[7].item() - expected_normalized_7) < 1e-4:
        print("✅ N-gram soft penalty applied correctly!")
    else:
        print(f"❌ N-gram penalty failed! Got {constrained_probs[7].item()}")

if __name__ == "__main__":
    run_verification()
