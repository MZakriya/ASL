import sys
import torch
import os

sys.path.append("d:\\All Projects\\ASLR")
from advanced_predict_translation import AdvancedTranslationPredictor

# Mock Model/Vocab
class MockModel:
    def eval(self): pass
    def encode(self, x): return x # Return dummy
    def decode(self, *args, **kwargs): return torch.zeros(1, 1, 512)
    def __call__(self, *args, **kwargs): return self

class MockVocab:
    def __len__(self): return 100
    def __getattr__(self, name): return 0

def test_kwargs():
    print("Testing Predict Kwargs...")
    
    # Config with Beam=1 (Greedy)
    config = {'beam_width': 1, 'max_length': 5, 'strict_repetition_penalty': 5.0, 'temperature': 0.4}
    
    predictor = AdvancedTranslationPredictor(MockModel(), MockVocab(), config)
    predictor.model.fc_out = torch.nn.Linear(512, 100) # Dummy head
    predictor.sos_id = 1
    predictor.eos_id = 2
    predictor.pad_id = 0
    
    # Mock _beam_search and _greedy_search to just print status
    # We can't easily mock methods on instance without monkeypatching class or instance
    # Instead, let's just run it and catch the print output or use a flag
    
    # Let's inspect config after call
    dummy_input = torch.randn(1, 10, 512)
    
    try:
        print("Calling predict(beam_width=5)...")
        # specific kwarg override
        predictor.predict(dummy_input, beam_width=5)
    except Exception as e:
        print(f"Error during call: {e}")
        
    print(f"Final Config Beam Info: {predictor.config.get('beam_width')}")
    
    if predictor.config.get('beam_width') == 5:
        print("SUCCESS: Config updated via kwargs.")
    else:
        print("FAILURE: Config NOT updated.")

if __name__ == "__main__":
    test_kwargs()
