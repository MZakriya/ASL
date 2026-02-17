import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, deque
import math
import logging

import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_debug(msg):
    with open("debug_logits.log", "a") as f:
        f.write(msg + "\n")

class AdvancedTranslationPredictor:
    """
    Standard Beam Search Predictor for SLT (v18 Model).
    Simplified to rely on model weights + Repetition Penalty.
    """

    def __init__(self, model, vocab, config=None):
        self.model = model
        self.vocab = vocab
        
        # Output config
        self.config = {
            'max_length': 12,
            'beam_width': 5,
            'strict_repetition_penalty': 50.0,
            'length_penalty_alpha': 0.7, # Updated to 0.7 per User Request
            'temperature': 0.5, # Updated to 0.5 per User Request (Decisive)
        }
        if config:
            self.config.update(config)
            
        # Hard-Fix Special Tokens per User Request (v18 Model Standards)
        self.pad_id = 0
        self.sos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        
        print(f"DEBUG: Hard-coded Special Tokens: PAD={self.pad_id}, SOS={self.sos_id}, EOS={self.eos_id}, UNK={self.unk_id}")
        
        # Verify against vocab if possible, but trust hardcodes for control flow
        if hasattr(self.vocab, 'id_to_token'):
            try:
                print(f"DEBUG: Vocab check - 1: {self.vocab.id_to_token(1)} / 2: {self.vocab.id_to_token(2)}")
            except:
                pass


    def predict(self, encoder_input, **kwargs):
        """
        Main prediction entry point.
        Allows overriding config keys via kwargs.
        """
        # Update config with runtime overrides
        initial_config = self.config.copy()
        for k, v in kwargs.items():
            self.config[k] = v
        
        self.model.eval()
        with torch.no_grad():
            # Encode
            encoder_output = self.model.encode(encoder_input)
            
        batch_size = encoder_input.size(0)
        results = []
        
        if self.config.get('beam_width', 1) == 1:
            print("DEBUG: Using strict Greedy Search per User Request.")
            best_seq = self._greedy_search(encoder_output)
            text = self._decode_sequence(best_seq['sequence'])
            best_seq['text'] = text
            results.append(best_seq)
            return results

        # Normalize penalty to be positive for subtraction
        if self.config['strict_repetition_penalty'] < 0:
             self.config['strict_repetition_penalty'] = abs(self.config['strict_repetition_penalty'])
        
        for i in range(batch_size):
            best_seq = self._beam_search(encoder_output[i:i+1])
            text = self._decode_sequence(best_seq['sequence'])
            best_seq['text'] = text
            results.append(best_seq)
            
        return results

    def _greedy_search(self, encoder_output):
        """
        User Request: Simple Greedy Search Baseline.
        STRICT ARGMAX for Alignment Verification.
        """
        max_len = self.config['max_length']
        # For alignment check, we might want to disable ALL penalties
        # But let's respect config if passed, default to 0 for strict check
        penalty = self.config.get('strict_repetition_penalty', 0.0) 
        temp = self.config.get('temperature', 1.0) # Default 1.0 for raw logits
        
        # Start with SOS
        ys = torch.ones(1, 1).fill_(self.sos_id).type_as(encoder_output).long()
        
        sequence = [self.sos_id]
        
        print(f"DEBUG: Starting Greedy Search (Penalty={penalty}, Temp={temp})")
        
        for i in range(max_len - 1):
            # Check mask
            tgt_mask = None
            if hasattr(self.model, 'generate_square_subsequent_mask'):
                 sz = ys.size(1)
                 tgt_mask = self.model.generate_square_subsequent_mask(sz).to(encoder_output.device)
            
            with torch.no_grad():
                out = self.model.decode(ys, encoder_output, tgt_mask)
                
            # Get last token logits
            last_hidden = out[:, -1, :] # [1, d_model]
            generator = getattr(self.model, 'fc_out', getattr(self.model, 'generator', None))
            logits = generator(last_hidden) # [1, vocab]
            
            # RAW LOGIT CHECK
            raw_val, raw_idx = torch.max(logits, dim=-1)
            raw_token = "UNK"
            if hasattr(self.vocab, 'id_to_token'): raw_token = self.vocab.id_to_token(raw_idx.item())
            elif hasattr(self.vocab, 'itos'): raw_token = self.vocab.itos.get(raw_idx.item(), "UNK")
            print(f"DEBUG: Step {i} RAW (No Pen): {raw_token} ({raw_idx.item()}) Val={raw_val.item():.4f}")

            # Apply Temp & Penalty (Strict)
            if temp != 1.0:
                logits = logits / (temp + 1e-9)
            
            # Repetition Penalty (Relaxed)
            if penalty > 0:
                for t in sequence:
                    if t not in [self.sos_id, self.eos_id, self.pad_id]:
                         logits[0, t] -= penalty
                     
            prob = torch.nn.functional.softmax(logits, dim=-1)
            next_word = torch.argmax(prob, dim=-1).item()
            
            # Debug Top 1 choice and Stats
            msg = f"Step {i}: Val={next_word} Prob={prob[0, next_word]:.4f} LogitMax={logits.max():.4f}"
            print(msg)
            # log_debug(msg) # Optional
            
            sequence.append(next_word)
            
            # Append to input
            ys = torch.cat([ys, torch.ones(1, 1).type_as(ys.data).fill_(next_word)], dim=1)
            
            if next_word == self.eos_id:
                break
                
        return {'sequence': sequence, 'score': 0.0, 'status': 'success'}
        
    def _beam_search(self, encoder_output):
        """
        Standard Beam Search with Repetition Penalty.
        """
        beam_width = self.config['beam_width']
        max_len = self.config['max_length']
        penalty = self.config['strict_repetition_penalty']
        temperature = self.config['temperature']
        
        # Beam: (sequence, score, used_tokens_set)
        # Sequence is list of token IDs
        start_seq = [self.sos_id]
        
        # List of beams
        beams = [(start_seq, 0.0, {self.sos_id})]
        
        for _ in range(max_len):
            candidates = []
            
            for seq, score, used_tokens in beams:
                if seq[-1] == self.eos_id:
                    # Completed
                    candidates.append((seq, score, used_tokens))
                    continue
                    
                # Get logits for next token
                # decoder_input: [1, len(seq)]
                dec_input = torch.tensor([seq], dtype=torch.long, device=encoder_output.device)
                
                # Decode with Mask (Fix for Loop Lock)
                tgt_mask = None
                if hasattr(self.model, 'generate_square_subsequent_mask'):
                    tgt_mask = self.model.generate_square_subsequent_mask(dec_input.size(1)).to(encoder_output.device)
                    
                with torch.no_grad():
                    try:
                        dec_out = self.model.decode(dec_input, encoder_output, tgt_mask=tgt_mask)
                    except TypeError:
                         # Fallback if model.decode doesn't accept tgt_mask yet (though we updated it)
                         dec_out = self.model.decode(dec_input, encoder_output)
                    
                # Project to vocab
                # dec_out: [1, seq_len, hidden] or [1, hidden]
                if dec_out.dim() == 3:
                    last_hidden = dec_out[:, -1, :]
                else:
                    last_hidden = dec_out
                    
                # Generator
                generator = getattr(self.model, 'fc_out', getattr(self.model, 'generator', None))
                logits = generator(last_hidden) # [1, vocab_size]
                
                # Apply Temperature (with safety)
                # User Critical Fix: Check for Division by Zero
                logits = logits / (temperature + 1e-9)
                
                # Check for NaNs
                if torch.isnan(logits).any():
                     print("CRITICAL WARNING: NaNs detected in logits! Using zero initialization.")
                     logits = torch.zeros_like(logits)
                     
                # DEBUG: Raw Logit Stats (Ensure no hidden bias)
                if len(seq) == 1 and len(beams) == 1:
                    print(f"DEBUG: Raw Logits Mean: {logits.mean().item():.4f}, Std: {logits.std().item():.4f}, Max: {logits.max().item():.4f}")

                # LOGIT DEBUGGING (User Request: Show words)
                if len(seq) < 4 and len(beams) == 1: # Only print for first beam
                     # Softmax Safety
                     probs_debug = torch.nn.functional.softmax(logits, dim=1)
                     topv, topi = torch.topk(probs_debug, 5)
                     
                     top_words = []
                     for tid in topi[0]:
                         tid_item = tid.item()
                         if hasattr(self.vocab, 'id_to_token'):
                             top_words.append(self.vocab.id_to_token(tid_item))
                         elif hasattr(self.vocab, 'itos'):
                             if isinstance(self.vocab.itos, list):
                                 top_words.append(self.vocab.itos[tid_item] if tid_item < len(self.vocab.itos) else str(tid_item))
                             else:
                                 # strict dict usage
                                 top_words.append(self.vocab.itos.get(tid_item, str(tid_item)))
                         else:
                             top_words.append(str(tid_item))
                             
                     print(f"DEBUG: Step {len(seq)} Top 5: {top_words} Probs: {topv[0].tolist()}")

                # Apply Repetition Penalty to Used Tokens (Relaxed)
                penalty = self.config.get('strict_repetition_penalty', 5.0)
                if penalty > 0:
                    # CUMULATIVE PENALTY via sequence iteration
                    for t in seq:
                        # Don't penalize EOS/PAD/SOS
                        if t not in [self.sos_id, self.eos_id, self.pad_id]:
                            if t < logits.shape[1]:
                                 logits[0, t] -= penalty
                                 
                # Log Softmax Safety
                log_probs = torch.nn.functional.log_softmax(logits, dim=1) # [1, vocab_size]
                
                # Top K
                topk_probs, topk_ids = torch.topk(log_probs, beam_width * 2)
                
                for k in range(beam_width * 2):
                     token = topk_ids[0, k].item()
                     lp = topk_probs[0, k].item()
                     
                     new_seq = seq + [token]
                     new_score = score + lp
                     new_used = used_tokens.copy()
                     new_used.add(token)
                     
                     candidates.append((new_seq, new_score, new_used))
            
            # Sort candidates by normalized score
            ordered = sorted(candidates, key=lambda x: self._score(x[1], len(x[0])), reverse=True)
            beams = ordered[:beam_width]
            
            # Check if all finished
            if all(b[0][-1] == self.eos_id for b in beams):
                break

                
        # Select best
        best = beams[0]
        return {
            'sequence': best[0],
            'score': best[1],
            'status': 'success'
        }
        
    def _score(self, log_prob, length):
        """Length matching normalization."""
        alpha = self.config['length_penalty_alpha']
        # Google NMT
        lp = ((5 + length) ** alpha) / ((5 + 1) ** alpha)
        return log_prob / lp

    def _decode_sequence(self, sequence):
        """Convert IDs to text."""
        words = []
        for idx in sequence:
            if idx in [self.sos_id, self.eos_id, self.pad_id]:
                continue
            
            word = ""
            if hasattr(self.vocab, 'id_to_token'):
                word = self.vocab.id_to_token(idx)
            elif hasattr(self.vocab, 'itos'):
                 if isinstance(self.vocab.itos, list) and idx < len(self.vocab.itos):
                     word = self.vocab.itos[idx]
                 else:
                     word = self.vocab.itos.get(idx, "")
            elif hasattr(self.vocab, 'index2word'):
                 word = self.vocab.index2word.get(idx, "")
            
            if word and word not in ['<sos>', '<eos>', '<pad>', '<unk>']:
                 words.append(word)
                 
        return " ".join(words)

# Dummy wrapper for compatibility if needed
def predict_translation(*args, **kwargs):
    pass