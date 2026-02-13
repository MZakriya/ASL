import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, deque
import math

def predict_translation(model,
                      encoder_input,
                      vocab,
                      max_length=100,
                      beam_width=5,
                      temperature=0.75,
                      length_penalty_alpha=0.6,
                      ngram_blocking_size=2,
                      frequency_penalty_weight=2.0,
                      attention_guided_weight=1.5):
    """
    Advanced prediction function with contextual constraints and inference-time heuristics
    to address mode collapse and repetitive loops in sign language translation.

    Args:
        model: Trained transformer model
        encoder_input: Input features (Holistic + I3D, shape: batch_size x seq_len x 2653)
        vocab: Vocabulary object with methods like token_to_id, id_to_token
        max_length: Maximum sequence length for generation
        beam_width: Width of beam search
        temperature: Softmax temperature for diversity control
        length_penalty_alpha: Alpha parameter for length-normalized beam search
        ngram_blocking_size: Size of n-grams to block (1 for unigrams, 2 for bigrams)
        frequency_penalty_weight: Weight for penalizing high-frequency bias words
        attention_guided_weight: Weight for feature-guided constraints

    Returns:
        List of translated sequences with highest scores
    """

    # Define high-frequency bias words to penalize
    bias_words = {'such', 'also', 'easily', 'research', 'should', 'an', 'the', 'a', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

    # Convert bias words to token IDs
    bias_word_ids = set()
    for word in bias_words:
        if hasattr(vocab, 'token_to_id') and word in vocab.token_to_id():
            bias_word_ids.add(vocab.token_to_id(word))
        elif hasattr(vocab, '__contains__') and word in vocab:
            # Assuming vocab supports direct lookup
            try:
                bias_word_ids.add(vocab[word])
            except:
                pass

    # Encode the input features
    with torch.no_grad():
        encoder_output = model.encode(encoder_input)  # [batch_size, src_seq_len, d_model]

    batch_size = encoder_input.size(0)
    vocab_size = len(vocab) if hasattr(vocab, '__len__') else getattr(vocab, 'size', 50000)  # fallback

    # Initialize beams for each batch
    beams = [[] for _ in range(batch_size)]

    # Start with initial state for each batch
    for b_idx in range(batch_size):
        # Initial state: [sequence_tokens, log_prob, ngram_history, attention_scores]
        initial_state = ([vocab.token_to_id('<sos>')], 0.0, [], [])
        beams[b_idx].append(initial_state)

    # Main decoding loop
    for step in range(max_length):
        new_beams = [[] for _ in range(batch_size)]

        for b_idx in range(batch_size):
            current_beams = beams[b_idx]

            # Collect all possible next states
            candidates = []

            for seq, log_prob, ngram_hist, attn_scores in current_beams:
                if len(seq) == 0:
                    continue

                # Prepare decoder input
                decoder_input = torch.tensor([seq], dtype=torch.long, device=encoder_input.device)

                # Forward through decoder
                with torch.no_grad():
                    decoder_output = model.decode(decoder_input, encoder_output[b_idx:b_idx+1])  # [1, tgt_seq_len, d_model]

                    # Get logits for the last position
                    last_logits = model.generator(decoder_output[:, -1, :])  # [1, vocab_size]

                    # Apply temperature scaling
                    scaled_logits = last_logits / temperature

                    # Apply frequency penalty to bias words
                    freq_penalty = torch.zeros_like(scaled_logits)
                    for bias_id in bias_word_ids:
                        if bias_id < freq_penalty.size(-1):
                            freq_penalty[0, bias_id] -= frequency_penalty_weight
                    scaled_logits += freq_penalty

                # Get probabilities
                probs = F.softmax(scaled_logits, dim=-1)[0]  # [vocab_size]

                # Get top-k candidates
                top_k_probs, top_k_indices = torch.topk(probs, beam_width * 2, dim=-1)

                # Generate candidate sequences
                for prob, token_id in zip(top_k_probs, top_k_indices):
                    token_id = token_id.item()
                    new_prob = log_prob + torch.log(prob + 1e-12).item()  # Add small epsilon to avoid log(0)

                    # Create new n-gram history for blocking
                    new_ngram_hist = ngram_hist.copy()
                    new_ngram_hist.append(token_id)

                    # Check for n-gram repetition
                    should_block = False
                    if len(new_ngram_hist) >= ngram_blocking_size:
                        # Check for n-gram repetition
                        current_ngram = tuple(new_ngram_hist[-ngram_blocking_size:])
                        prev_ngrams = [tuple(new_ngram_hist[i:i+ngram_blocking_size])
                                      for i in range(len(new_ngram_hist)-ngram_blocking_size+1)]

                        # Count occurrences of current n-gram
                        ngram_count = sum(1 for ng in prev_ngrams if ng == current_ngram)
                        if ngram_count > 1:
                            should_block = True

                    # Check for end-of-sequence token
                    eos_token_id = vocab.token_to_id('<eos>') if hasattr(vocab, 'token_to_id') else 2  # Common EOS token id
                    if token_id == eos_token_id:
                        # Add to final candidates with length normalization
                        normalized_score = new_prob / ((len(seq) + 1) ** length_penalty_alpha)
                        candidates.append((seq + [token_id], normalized_score, new_ngram_hist, attn_scores))
                        continue

                    if not should_block:
                        # Apply feature-guided constraint based on attention
                        # Calculate similarity between predicted token and encoder features
                        attention_guided_score = 0

                        if hasattr(model, 'calculate_attention_alignment'):
                            # If model has a method to calculate attention alignment
                            try:
                                alignment_score = model.calculate_attention_alignment(
                                    encoder_output[b_idx], decoder_output[0, -1], token_id
                                )
                                attention_guided_score = alignment_score * attention_guided_weight
                            except:
                                # Fallback: use simple heuristic based on encoder features
                                encoder_features = encoder_output[b_idx].mean(dim=0)  # Average encoder features
                                # This is a simplified version - in practice, you'd use actual attention weights
                                attention_guided_score = 0.0

                        adjusted_prob = new_prob + attention_guided_score

                        candidates.append((seq + [token_id], adjusted_prob, new_ngram_hist, attn_scores))

            # Sort candidates by probability and keep top beam_width
            candidates.sort(key=lambda x: x[1], reverse=True)
            new_beams[b_idx] = candidates[:beam_width]

        beams = new_beams

        # Early stopping: if all beams have generated EOS, break
        all_finished = True
        for b_idx in range(batch_size):
            if beams[b_idx]:
                # Check if all current beams have EOS
                for seq, _, _, _ in beams[b_idx]:
                    eos_token_id = vocab.token_to_id('<eos>') if hasattr(vocab, 'token_to_id') else 2
                    if eos_token_id not in seq:
                        all_finished = False
                        break
                if not all_finished:
                    break
            else:
                all_finished = False

        if all_finished:
            break

    # Select best translation for each batch
    results = []
    for b_idx in range(batch_size):
        if beams[b_idx]:
            # Sort by normalized score and get best
            best_seq, best_score, _, _ = max(beams[b_idx], key=lambda x: x[1])
            results.append({
                'sequence': best_seq,
                'score': best_score,
                'text': ' '.join([vocab.id_to_token(token_id) if hasattr(vocab, 'id_to_token')
                                 else str(token_id) for token_id in best_seq[1:-1]])  # Exclude SOS and EOS
            })
        else:
            # Fallback if no valid sequence found
            results.append({
                'sequence': [vocab.token_to_id('<sos>'), vocab.token_to_id('<eos>')],
                'score': float('-inf'),
                'text': ''
            })

    return results


def calculate_ngram_repetition_penalty(sequence, n=2):
    """
    Calculate penalty for n-gram repetitions in the sequence
    """
    if len(sequence) < n:
        return 0.0

    ngrams = {}
    for i in range(len(sequence) - n + 1):
        ngram = tuple(sequence[i:i+n])
        ngrams[ngram] = ngrams.get(ngram, 0) + 1

    # Calculate penalty based on repetitions
    penalty = 0.0
    for ngram, count in ngrams.items():
        if count > 1:
            penalty += (count - 1) * 0.5  # Penalty increases with repetition count

    return penalty


def apply_length_normalization(score, length, alpha=0.6):
    """
    Apply length normalization to the score
    """
    # Common length normalization: score / (length ^ alpha)
    return score / (length ** alpha)


def get_cross_attention_scores(encoder_features, decoder_state, model):
    """
    Calculate cross-attention scores between encoder features and decoder state
    This helps ensure feature-guided word selection
    """
    if hasattr(model, 'cross_attention_layer'):
        # Use the model's cross-attention layer to compute alignment
        with torch.no_grad():
            attention_weights = model.cross_attention_layer(
                decoder_state.unsqueeze(0),
                encoder_features.unsqueeze(0)
            )[0]  # Remove batch dimension
            return attention_weights.mean().item()  # Average attention weight as alignment score
    else:
        # Fallback: use cosine similarity
        encoder_mean = encoder_features.mean(dim=0)
        cos_sim = F.cosine_similarity(decoder_state.unsqueeze(0), encoder_mean.unsqueeze(0))
        return cos_sim.item()


# Alternative implementation with more detailed control
def predict_translation_advanced(model,
                               encoder_input,
                               vocab,
                               max_length=100,
                               beam_width=5,
                               temperature=0.75,
                               **kwargs):
    """
    Advanced version with more modular components
    """

    # Configuration
    config = {
        'length_penalty_alpha': kwargs.get('length_penalty_alpha', 0.6),
        'ngram_blocking_size': kwargs.get('ngram_blocking_size', 2),
        'frequency_penalty_weight': kwargs.get('frequency_penalty_weight', 2.0),
        'attention_guided_weight': kwargs.get('attention_guided_weight', 1.5),
        'bias_words': kwargs.get('bias_words',
                               {'such', 'also', 'easily', 'research', 'should', 'an', 'the', 'a'})
    }

    return predict_translation(model, encoder_input, vocab, max_length, beam_width,
                             temperature, **config)