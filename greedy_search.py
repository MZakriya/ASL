"""
Greedy Search with Teacher Forcing

This module provides a simple greedy search decoder with temperature 0.1
for debugging semantic alignment issues. Use this to see if the model can
produce coherent output without beam search complexity.

Usage:
    from greedy_search import greedy_decode
    
    result = greedy_decode(
        model=model,
        encoder_input=input_tensor,
        vocab=vocab,
        max_length=20,
        temperature=0.1
    )
"""

import torch
import torch.nn.functional as F

def greedy_decode(model, encoder_input, vocab, max_length=20, temperature=0.8):
    """
    Greedy decoding with low temperature for coherent output
    
    Args:
        model: Transformer model
        encoder_input: Input tensor [batch, seq_len, features]
        vocab: Vocabulary object
        max_length: Maximum sequence length
        temperature: Sampling temperature (0.1 for near-deterministic)
    
    Returns:
        dict with 'text', 'sequence', 'confidence'
    """
    model.eval()
    device = encoder_input.device
    
    # Encode
    with torch.no_grad():
        encoder_output = model.encode(encoder_input)
    
    # Get special token IDs
    if hasattr(vocab, 'token_to_id'):
        sos_id = vocab.token_to_id('<sos>') or vocab.token_to_id('<SOS>') or 1
        eos_id = vocab.token_to_id('<eos>') or vocab.token_to_id('<EOS>') or 2
        pad_id = vocab.token_to_id('<pad>') or vocab.token_to_id('<PAD>') or 0
    else:
        sos_id = vocab.stoi.get('<SOS>', vocab.stoi.get('<sos>', 1))
        eos_id = vocab.stoi.get('<EOS>', vocab.stoi.get('<eos>', 2))
        pad_id = vocab.stoi.get('<PAD>', vocab.stoi.get('<pad>', 0))
    
    # Initialize sequence
    sequence = [sos_id]
    token_probs = []
    
    print(f"\n{'='*70}")
    print("GREEDY SEARCH (Temperature {:.1f})".format(temperature))
    print(f"{'='*70}")
    
    for step in range(max_length):
        # Prepare decoder input
        decoder_input = torch.tensor([sequence], dtype=torch.long, device=device)
        
        # Decode
        with torch.no_grad():
            decoder_output = model.decode(decoder_input, encoder_output)
            
            # Get logits for last position
            if hasattr(model, 'generator'):
                logits = model.generator(decoder_output[:, -1, :])
            elif hasattr(model, 'fc_out'):
                logits = model.fc_out(decoder_output[:, -1, :])
            else:
                raise ValueError("Model has no generator or fc_out layer")
        
        # Apply temperature
        logits = logits / temperature
        
        
        # Get probabilities
        probs = F.softmax(logits, dim=-1).squeeze(0)
        
        # CRITICAL: Strict Repetition Penalty (Mode Collapse Fix)
        # If a word is predicted once, reduce its probability by 90% for next 5 steps
        if len(sequence) > 1:
            # Get last 5 tokens (or fewer if sequence is shorter)
            lookback = min(5, len(sequence) - 1)
            recent_tokens = sequence[-lookback:]
            
            for prev_token in recent_tokens:
                if prev_token != sos_id:  # Don't penalize SOS
                    # 90% reduction as requested
                    probs[prev_token] *= 0.10
        
        # Renormalize after penalties
        probs = probs / probs.sum()
        
        # Greedy selection: take argmax
        next_token_id = torch.argmax(probs).item()
        next_token_prob = probs[next_token_id].item()
        
        # Get word
        if hasattr(vocab, 'id_to_token'):
            word = vocab.id_to_token(next_token_id)
        elif hasattr(vocab, 'itos'):
            if isinstance(vocab.itos, list):
                word = vocab.itos[next_token_id] if next_token_id < len(vocab.itos) else '<UNK>'
            else:
                word = vocab.itos.get(next_token_id, '<UNK>')
        else:
            word = f'<ID_{next_token_id}>'
        
        print(f"Step {step:2d}: Token {next_token_id:4d} = '{word:15s}' (prob: {next_token_prob:.4f})")
        
        # Add to sequence
        sequence.append(next_token_id)
        token_probs.append(next_token_prob)
        
        # Check for EOS
        if next_token_id == eos_id:
            print(f"[EOS reached at step {step}]")
            break
    
    # Decode sequence to text
    words = []
    for token_id in sequence[1:]:  # Skip SOS
        if token_id == eos_id:
            break
        if token_id == pad_id:
            continue
        
        if hasattr(vocab, 'id_to_token'):
            word = vocab.id_to_token(token_id)
        elif hasattr(vocab, 'itos'):
            if isinstance(vocab.itos, list):
                word = vocab.itos[token_id] if token_id < len(vocab.itos) else '<UNK>'
            else:
                word = vocab.itos.get(token_id, '<UNK>')
        else:
            word = f'<ID_{token_id}>'
        
        if not word.startswith('<'):
            words.append(word)
    
    text = ' '.join(words)
    avg_confidence = sum(token_probs) / len(token_probs) if token_probs else 0.0
    
    print(f"\nGreedy Output: {text}")
    print(f"Average Confidence: {avg_confidence:.4f}")
    print(f"{'='*70}\n")
    
    return {
        'text': text,
        'sequence': sequence,
        'confidence': avg_confidence,
        'token_probs': token_probs
    }
