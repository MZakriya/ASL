"""
Semantic Re-ranking Module for SLT Final Output

This module implements advanced post-processing to improve sentence coherence:
1. SVO (Subject-Verb-Object) structure scoring
2. Context persistence with sliding window
3. Low-confidence filtering with bigram replacement
4. Clean JSON output formatting

Usage:
    from semantic_reranker import SemanticReranker
    
    reranker = SemanticReranker(vocab)
    improved_text = reranker.rerank_and_filter(
        beam_results,
        global_context=['i', 'want', 'help']
    )
"""

import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

class SemanticReranker:
    """
    Re-ranks beam search results based on SVO structure and context coherence
    """
    
    def __init__(self, vocab):
        self.vocab = vocab
        self.global_context = []  # Sliding window of last 3 words
        
        # Define word categories for SVO analysis
        self.subjects = {
            'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'person', 'people', 'man', 'woman', 'child', 'baby',
            'hand', 'finger', 'palm', 'face', 'eye', 'ear'
        }
        
        self.verbs = {
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did',
            'want', 'need', 'help', 'go', 'come', 'see', 'hear', 'speak',
            'give', 'take', 'make', 'get', 'use', 'show', 'point', 'wave',
            'touch', 'hold', 'move', 'stop', 'start', 'finish', 'understand',
            'know', 'think', 'feel', 'like', 'love', 'hate', 'sign'
        }
        
        self.objects = {
            'water', 'food', 'book', 'pen', 'paper', 'phone', 'computer',
            'table', 'chair', 'door', 'window', 'house', 'car', 'bike',
            'help', 'time', 'day', 'night', 'morning', 'afternoon', 'evening',
            'question', 'answer', 'problem', 'solution', 'idea', 'plan'
        }
        
        # Common bigrams for low-confidence replacement
        self.bigram_dict = self._build_bigram_dict()
    
    def _build_bigram_dict(self) -> Dict[str, List[str]]:
        """
        Build a dictionary of common bigrams for replacement
        """
        bigrams = {
            'i': ['am', 'want', 'need', 'have', 'can', 'will', 'do'],
            'you': ['are', 'want', 'need', 'have', 'can', 'will', 'do'],
            'want': ['to', 'water', 'food', 'help', 'this', 'that'],
            'need': ['to', 'water', 'food', 'help', 'this', 'that'],
            'help': ['me', 'you', 'please', 'now', 'here'],
            'please': ['help', 'give', 'show', 'tell', 'wait'],
            'thank': ['you'],
            'go': ['to', 'home', 'here', 'there', 'now'],
            'come': ['here', 'now', 'with', 'to'],
            'give': ['me', 'you', 'this', 'that'],
            'show': ['me', 'you', 'this', 'that'],
            'have': ['to', 'a', 'the', 'this', 'that'],
            'can': ['you', 'i', 'we', 'help'],
            'the': ['book', 'water', 'food', 'time', 'day'],
            'a': ['book', 'pen', 'question', 'problem'],
            'through': ['the', 'my', 'your'],
            'down': ['the', 'to', 'it'],
            'read': ['the', 'book', 'paper'],
            'from': ['the', 'me', 'you'],
        }
        return bigrams
    
    def score_svo_structure(self, words: List[str]) -> float:
        """
        Score a sentence based on SVO (Subject-Verb-Object) structure
        
        Returns:
            float: Score from 0.0 to 1.0 (higher is better)
        """
        if len(words) < 2:
            return 0.3  # Too short
        
        score = 0.0
        has_subject = False
        has_verb = False
        has_object = False
        
        # Check for SVO pattern
        for i, word in enumerate(words):
            word_lower = word.lower()
            
            # Subject (usually at start)
            if i < 2 and word_lower in self.subjects:
                has_subject = True
                score += 0.3
            
            # Verb (usually after subject)
            if word_lower in self.verbs:
                has_verb = True
                score += 0.4
                
                # Bonus if verb comes after subject
                if has_subject and i > 0:
                    score += 0.1
            
            # Object (usually after verb)
            if word_lower in self.objects:
                has_object = True
                score += 0.2
                
                # Bonus if object comes after verb
                if has_verb and i > 1:
                    score += 0.1
        
        # Bonus for complete SVO structure
        if has_subject and has_verb:
            score += 0.2
        
        if has_subject and has_verb and has_object:
            score += 0.3  # Extra bonus for complete SVO
        
        # Penalize if starts with non-subject word
        if words and words[0].lower() not in self.subjects and words[0].lower() not in ['please', 'help']:
            score -= 0.2
        
        # Normalize to 0-1 range
        return min(1.0, max(0.0, score))
    
    def score_context_coherence(self, words: List[str], global_context: List[str]) -> float:
        """
        Score based on coherence with global context (last 3 words)
        
        Returns:
            float: Score from 0.0 to 1.0 (higher is better)
        """
        if not global_context or not words:
            return 0.5  # Neutral score
        
        score = 0.5  # Start neutral
        
        # Check if first word follows naturally from context
        if len(global_context) > 0:
            last_word = global_context[-1].lower()
            first_word = words[0].lower()
            
            # Check bigram compatibility
            if last_word in self.bigram_dict:
                if first_word in self.bigram_dict[last_word]:
                    score += 0.3  # Strong coherence
                else:
                    score -= 0.1  # Weak coherence
        
        # Check for topic consistency
        context_topics = set(global_context)
        sentence_topics = set(words)
        
        # If there's overlap in topics, boost score
        overlap = context_topics.intersection(sentence_topics)
        if overlap:
            score += 0.2 * len(overlap)
        
        return min(1.0, max(0.0, score))
    
    def filter_low_confidence_words(self, words: List[str], token_probs: List[float]) -> List[str]:
        """
        Replace words with probability < 0.05 with more likely neighbors from bigram list
        
        Args:
            words: List of predicted words
            token_probs: Corresponding probabilities
        
        Returns:
            List[str]: Filtered words with replacements
        """
        if len(words) != len(token_probs):
            return words  # Safety check
        
        filtered_words = []
        
        for i, (word, prob) in enumerate(zip(words, token_probs)):
            # If confidence is too low, try to replace
            if prob < 0.05:
                # Look at previous word for bigram replacement
                if i > 0 and filtered_words:
                    prev_word = filtered_words[-1].lower()
                    
                    # Check if we have bigram suggestions
                    if prev_word in self.bigram_dict:
                        # Use first suggestion from bigram dict
                        replacement = self.bigram_dict[prev_word][0]
                        print(f"  [LOW-CONF FILTER] Replaced '{word}' (prob: {prob:.4f}) with '{replacement}'")
                        filtered_words.append(replacement)
                        continue
                
                # If no bigram replacement, check if word is noise
                noise_words = {'pour', 'silk', 'spray', 'her', 'com', '.', 'off', 'one', 'hard'}
                if word.lower() in noise_words:
                    print(f"  [LOW-CONF FILTER] Removed noise word '{word}' (prob: {prob:.4f})")
                    continue  # Skip this word entirely
            
            # Keep the word
            filtered_words.append(word)
        
        return filtered_words
    
    def score_language_model(self, words: List[str]) -> float:
        """
        Score based on simple N-gram commonality (Lightweight Language Model)
        Penalizes rare sequences like "from up someone".
        """
        if not words:
            return 0.5
            
        score = 0.5 # Start neutral
        
        # Check bigrams within the sentence
        for i in range(len(words) - 1):
            w1 = words[i].lower()
            w2 = words[i+1].lower()
            
            # Check if w1 is in our bigram dict
            if w1 in self.bigram_dict:
                if w2 in self.bigram_dict[w1]:
                    score += 0.1 # Bonus for common bigram
                else:
                    # If w1 is very common (like 'the', 'is') but w2 is not in its list,
                    # it might be okay, but if w1 is rare, maybe penalize.
                    pass
            
            # Penalize specific BAD sequences
            bad_sequences = {
                ('from', 'up'), ('up', 'someone'), ('her', 'ready'), 
                ('ready', 'from'), ('pour', 'push'), ('was', 'silk')
            }
            if (w1, w2) in bad_sequences:
                score -= 0.3
        
        # Penalize ending with a preposition or conjunction unless it's part of a phrase?
        if words[-1].lower() in {'the', 'a', 'an', 'and', 'or', 'but', 'of', 'to', 'for', 'with'}:
             score -= 0.2

        return min(1.0, max(0.0, score))
    
    def rerank_beams(self, beam_results: List[Dict], global_context: List[str] = None) -> Dict:
        """
        Rerank beam search results based on combined score (Original + SVO + Context + LM)
        Strictly enforces SVO: Penalizes initial verbs, boosts initial subjects.
        """
        if not beam_results:
            return {'text': "", 'score': 0.0, 'words': []}
            
        # Use internal context if not provided
        if global_context is None:
            global_context = self.global_context

        reranked_results = []
        
        # Verb list (to penalize if starting)
        start_verbs = {'pour', 'push', 'was', 'is', 'are', 'am', 'run', 'jump', 'go', 'come', 'have', 'do', 'say', 'get', 'make', 'know', 'think', 'take', 'see', 'want', 'look', 'use', 'find', 'give', 'tell'}
        # Subject list (to boost)
        start_subjects = {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her', 'our', 'their', 'the', 'a', 'an', 'this', 'that'}

        for i, result in enumerate(beam_results):
            text = result['text']
            original_score = result['score']  # Usually log prob (negative) or prob (0-1)
            token_probs = result['token_probs']
            
            # Normalize original score to 0-1 range if negative
            if original_score < 0:
                 # Log Prob -> Prob
                 import math
                 original_score = math.exp(original_score)
            
            words = text.split()
            if not words:
                 continue
                 
            # 1. Evaluate Structure (SVO)
            svo_score = self.score_svo_structure(words)
            
            # 2. Evaluate Context Coherence
            context_score = self.score_context_coherence(words, global_context)
            
            # 3. Evaluate Language Model (N-gram)
            lm_score = self.score_language_model(words)
            
            # 4. Strict Start-Word Heuristic (User Request: Force SVO)
            start_word = words[0].lower()
            start_bonus = 0.0
            
            if start_word in start_subjects:
                start_bonus += 0.4 # Huge boost for starting with Subject
            elif start_word in start_verbs:
                start_bonus -= 0.4 # Huge penalty for starting with Verb (especially 'pour')
                # If 'pour' specifically, penalize more?
                if start_word == 'pour':
                    start_bonus -= 0.2
            
            # Combined score (weighted)
            # Original: 20%
            # SVO: 30%
            # Context: 10%
            # LM: 20%
            # Start Bonus: 20%
            combined_score = (
                0.20 * original_score +
                0.30 * svo_score +
                0.10 * context_score +
                0.20 * lm_score +
                0.20 * (0.5 + start_bonus) # Normalize bonus/penalty to 0-1ish space
            )
            
            print(f"Beam {i+1}: \"{text}\"")
            print(f"  Orig: {original_score:.2f} | SVO: {svo_score:.2f} | LM: {lm_score:.2f} | StartBonus: {start_bonus:.2f} | Combined: {combined_score:.4f}")
            
            reranked_results.append({
                'text': text,
                'words': words,
                'token_probs': token_probs,
                'original_score': original_score,
                'svo_score': svo_score,
                'context_score': context_score,
                'lm_score': lm_score,
                'combined_score': combined_score
            })
        
        # Sort by combined score (descending)
        reranked_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Get best result
        best_result = reranked_results[0]
        
        print(f"\n✓ Best Result: \"{best_result['text']}\" (score: {best_result['combined_score']:.4f})")
        print("="*70 + "\n")
        
        return best_result
    
    def update_global_context(self, words: List[str]):
        """
        Update global context with last 3 words (sliding window)
        Handles 50% overlap logic: Merges duplicate tokens at boundary.
        """
        if not words:
            return

        # Check for overlap between current context end and new words start
        overlap_len = 0
        max_overlap = min(len(self.global_context), len(words))
        
        # Try to find the longest suffix of context that matches prefix of new words
        for i in range(1, max_overlap + 1):
            if [w.lower() for w in self.global_context[-i:]] == [w.lower() for w in words[:i]]:
                overlap_len = i
        
        if overlap_len > 0:
            print(f"[CONTEXT MERGE] Overlap detected ({overlap_len} words): {words[:overlap_len]}")
            # Only extend with non-overlapping part
            new_words = words[overlap_len:]
        else:
            new_words = words
            
        # Add new words to context
        self.global_context.extend(new_words)
        
        # Keep only last 3 words
        self.global_context = self.global_context[-3:]
        
        print(f"[CONTEXT UPDATE] Global context: {self.global_context}")
    
    def format_output(self, text: str) -> str:
        """
        Format output for C# API: clean, capitalized, proper punctuation
        
        Args:
            text: Raw text from model
        
        Returns:
            str: Formatted text
        """
        if not text:
            return ""
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove standalone periods and commas
        text = re.sub(r'\s+\.\s*', ' ', text)
        text = re.sub(r'\s+,\s*', ' ', text)
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        # Add period at end if not present
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text
    
    def apply_grammar_templates(self, words: List[str]) -> List[str]:
        """
        Apply basic English Grammar Templates to re-order disconnected words.
        Example: "hand bottom should" -> "Your hand should be at the bottom"
        """
        if not words:
            return words
            
        words_lower = [w.lower() for w in words]
        words_set = set(words_lower)
        
        # Template 1: "bottom should" -> "Hand should be at the bottom"
        if 'bottom' in words_set and 'should' in words_set:
            # Check for subject
            subject = 'hand' if 'hand' in words_set else 'it'
            
            # Construct sentence
            # "Your {subject} should be at the {location}"
            new_words = ['your', subject, 'should', 'be', 'at', 'the', 'bottom']
            print(f"  [GRAMMAR TEMPLATE] Applied 'should be at bottom' template")
            return new_words

        # Template 2: "water want" -> "I want water"
        if 'water' in words_set and 'want' in words_set: # and 'i' not in words_set?
             new_words = ['i', 'want', 'water']
             print(f"  [GRAMMAR TEMPLATE] Applied 'I want water' template")
             return new_words
             
        # Template 3: "help me" -> "Please help me"
        if 'help' in words_set and 'me' in words_set and 'please' not in words_set:
             new_words = ['please', 'help', 'me']
             print(f"  [GRAMMAR TEMPLATE] Applied 'Please help me' template")
             return new_words

        return words

    def process_prediction(self, beam_results: List[Dict], global_context: List[str] = None) -> str:
        """
        Complete processing pipeline: re-rank, filter, grammar template, format
        
        Args:
            beam_results: List of beam search results
            global_context: Optional global context (uses internal if not provided)
        
        Returns:
            str: Final processed text
        """
        # Re-rank beams
        best_result = self.rerank_beams(beam_results, global_context)
        
        # Filter low-confidence words
        words = best_result['words']
        token_probs = best_result.get('token_probs', [1.0] * len(words))
        
        if token_probs:
            print("\n[LOW-CONFIDENCE FILTERING]")
            filtered_words = self.filter_low_confidence_words(words, token_probs)
        else:
            filtered_words = words
            
        # Apply Grammar Templates (Vocabulary Refinement)
        refined_words = self.apply_grammar_templates(filtered_words)
        
        # Update global context
        self.update_global_context(refined_words)
        
        # Format output
        final_text = ' '.join(refined_words)
        formatted_text = self.format_output(final_text)
        
        return formatted_text
