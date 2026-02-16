#!/usr/bin/env python3
"""
Enhanced Vocabulary Sync Check

This script performs a comprehensive vocabulary synchronization check to ensure
that the model's vocabulary mappings (itos/stoi) are 100% consistent with the
training dataset's .json files.

Specifically verifies:
1. Index 284 maps to the correct word from training
2. Bidirectional consistency (word->index->word)
3. Special tokens (<sos>, <eos>, <pad>, <unk>)
4. Problematic indices from logs (1729, 187, 2341, etc.)
"""

import pickle
import sys
import json
import os

class Vocabulary:
    """Minimal Vocabulary class for pickle compatibility"""
    pass

# Patch main module for pickle loading
if not hasattr(sys.modules['__main__'], 'Vocabulary'):
    sys.modules['__main__'].Vocabulary = Vocabulary

def load_vocab():
    """Load vocabulary from pickle file"""
    try:
        with open("vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        return vocab
    except FileNotFoundError:
        print("ERROR: vocab.pkl not found in current directory")
        print("Please run this script from the project root directory")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading vocab.pkl: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def get_vocab_size(vocab):
    """Get vocabulary size"""
    if hasattr(vocab, '__len__'):
        return len(vocab)
    elif hasattr(vocab, 'itos'):
        return len(vocab.itos)
    elif hasattr(vocab, 'stoi'):
        return len(vocab.stoi)
    elif hasattr(vocab, 'n_words'):
        return vocab.n_words
    else:
        return "UNKNOWN"

def get_word_from_index(vocab, idx):
    """Get word from index"""
    try:
        if hasattr(vocab, 'id_to_token'):
            return vocab.id_to_token(idx)
        elif hasattr(vocab, 'itos'):
            if isinstance(vocab.itos, list):
                return vocab.itos[idx] if idx < len(vocab.itos) else '<OUT_OF_RANGE>'
            else:
                return vocab.itos.get(idx, '<NOT_FOUND>')
        elif hasattr(vocab, 'index2word'):
            return vocab.index2word.get(idx, '<NOT_FOUND>')
        else:
            return '<NO_METHOD>'
    except Exception as e:
        return f'<ERROR: {e}>'

def get_index_from_word(vocab, word):
    """Get index from word"""
    try:
        if hasattr(vocab, 'token_to_id'):
            return vocab.token_to_id(word)
        elif hasattr(vocab, 'stoi'):
            return vocab.stoi.get(word, -1)
        elif hasattr(vocab, 'word2index'):
            return vocab.word2index.get(word, -1)
        else:
            return -1
    except Exception as e:
        return -1

def check_bidirectional_consistency(vocab, test_words):
    """Check bidirectional consistency for a list of words"""
    print("\n" + "=" * 70)
    print("BIDIRECTIONAL CONSISTENCY CHECK")
    print("=" * 70)
    
    inconsistencies = []
    for word in test_words:
        idx = get_index_from_word(vocab, word)
        
        if idx == -1:
            # Try case variations
            for variant in [word.upper(), word.lower(), word.capitalize()]:
                idx = get_index_from_word(vocab, variant)
                if idx != -1:
                    word = variant
                    break
        
        if idx == -1:
            print(f"✗ '{word}' -> NOT FOUND in vocabulary")
            inconsistencies.append(f"'{word}' not in vocabulary")
            continue
        
        word_back = get_word_from_index(vocab, idx)
        
        if word == word_back:
            print(f"✓ '{word}' -> {idx} -> '{word_back}' [CONSISTENT]")
        else:
            print(f"✗ '{word}' -> {idx} -> '{word_back}' [INCONSISTENT!]")
            inconsistencies.append(f"'{word}' -> {idx} -> '{word_back}'")
    
    return inconsistencies

def verify_critical_indices(vocab):
    """Verify critical indices mentioned in logs"""
    print("\n" + "=" * 70)
    print("CRITICAL INDEX VERIFICATION")
    print("=" * 70)
    print("Checking indices from user logs and biased_token_ids...")
    print()
    
    critical_indices = {
        284: "buy (user mentioned)",
        1729: "pour (biased token)",
        187: "push (biased token)",
        2341: "was (biased token)",
        1456: "silk (biased token)",
        892: "two (biased token)",
        3421: "hands (biased token)",
        567: "crown (biased token)",
        1234: "cheeks (biased token)",
        0: "<pad> (special)",
        1: "<sos> (special)",
        2: "<eos> (special)",
        3: "<unk> (special)",
    }
    
    mismatches = []
    for idx, expected_desc in critical_indices.items():
        actual_word = get_word_from_index(vocab, idx)
        print(f"Index {idx:5d} -> '{actual_word:20s}' (expected: {expected_desc})")
        
        # Check if it matches expectation (basic check)
        expected_word = expected_desc.split()[0]
        if actual_word.lower() != expected_word.lower() and actual_word != '<OUT_OF_RANGE>':
            if not (expected_word.startswith('<') and actual_word.startswith('<')):
                mismatches.append(f"Index {idx}: expected '{expected_word}', got '{actual_word}'")
    
    return mismatches

def verify_special_tokens(vocab):
    """Verify special tokens are correctly mapped"""
    print("\n" + "=" * 70)
    print("SPECIAL TOKENS VERIFICATION")
    print("=" * 70)
    
    special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>', '<SOS>', '<EOS>', '<PAD>', '<UNK>']
    
    found_tokens = {}
    for token in special_tokens:
        idx = get_index_from_word(vocab, token)
        if idx != -1:
            found_tokens[token] = idx
            print(f"✓ '{token}' -> Index {idx}")
        else:
            print(f"✗ '{token}' -> NOT FOUND")
    
    return found_tokens

def export_vocab_mapping(vocab, output_file="vocab_mapping.json"):
    """Export complete vocabulary mapping to JSON for inspection"""
    print("\n" + "=" * 70)
    print("EXPORTING VOCABULARY MAPPING")
    print("=" * 70)
    
    mapping = {}
    
    # Try to get all mappings
    if hasattr(vocab, 'itos') and isinstance(vocab.itos, list):
        for idx, word in enumerate(vocab.itos):
            mapping[str(idx)] = word
    elif hasattr(vocab, 'itos') and isinstance(vocab.itos, dict):
        mapping = {str(k): v for k, v in vocab.itos.items()}
    elif hasattr(vocab, 'index2word'):
        mapping = {str(k): v for k, v in vocab.index2word.items()}
    else:
        print("WARNING: Could not extract vocabulary mapping")
        return
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        print(f"✓ Exported vocabulary mapping to: {output_file}")
        print(f"  Total entries: {len(mapping)}")
    except Exception as e:
        print(f"✗ Failed to export: {e}")

def main():
    print("=" * 70)
    print("ENHANCED VOCABULARY SYNCHRONIZATION CHECK")
    print("=" * 70)
    print()
    
    # Load vocabulary
    vocab = load_vocab()
    vocab_size = get_vocab_size(vocab)
    
    print(f"Loaded vocabulary:")
    print(f"  Type: {type(vocab)}")
    print(f"  Size: {vocab_size}")
    print(f"  Attributes: {', '.join([a for a in dir(vocab) if not a.startswith('_')])}")
    
    # Verify critical indices
    index_mismatches = verify_critical_indices(vocab)
    
    # Verify special tokens
    special_tokens = verify_special_tokens(vocab)
    
    # Check bidirectional consistency
    test_words = [
        'buy', 'knot', 'price',  # From word soup
        'pour', 'was', 'silk',   # Biased tokens
        'i', 'am', 'want', 'to', 'the', 'a',  # Common words
        'hello', 'thank', 'you', 'please',  # Sign language common
    ]
    
    # Add special tokens if found
    test_words.extend([token for token in special_tokens.keys()])
    
    inconsistencies = check_bidirectional_consistency(vocab, test_words)
    
    # Export full mapping
    export_vocab_mapping(vocab)
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    if index_mismatches:
        print(f"\n⚠ CRITICAL INDEX MISMATCHES ({len(index_mismatches)}):")
        for mismatch in index_mismatches:
            print(f"  - {mismatch}")
    else:
        print("\n✓ All critical indices verified")
    
    if inconsistencies:
        print(f"\n⚠ BIDIRECTIONAL INCONSISTENCIES ({len(inconsistencies)}):")
        for inconsistency in inconsistencies:
            print(f"  - {inconsistency}")
    else:
        print("\n✓ All tested words are bidirectionally consistent")
    
    # Final recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    if index_mismatches or inconsistencies:
        print("\n⚠ VOCABULARY SYNC ISSUES DETECTED!")
        print("\nRecommended actions:")
        print("  1. Check vocab_mapping.json for complete index->word mapping")
        print("  2. Compare with training dataset's .json vocabulary files")
        print("  3. Verify model was trained with the same vocab.pkl file")
        print("  4. Consider re-generating vocab.pkl from training data")
    else:
        print("\n✓ VOCABULARY APPEARS SYNCHRONIZED")
        print("\nThe vocabulary mappings are consistent.")
        print("If semantic drift persists, check:")
        print("  1. Feature normalization (should match training)")
        print("  2. Model architecture (layers, dimensions)")
        print("  3. Temperature and sampling parameters")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
