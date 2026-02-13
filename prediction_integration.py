"""
Utility functions for integrating the advanced prediction system with your SLT model
"""

import torch
import torch.nn.functional as F
import advanced_predict_translation
from advanced_predict_translation import predict_translation, AdvancedTranslationPredictor
import logging

logger = logging.getLogger(__name__)

def integrate_with_model(model, encoder_input, vocab, **kwargs):
    """
    Integrate the advanced prediction system with your existing model

    Args:
        model: Your trained SLT model
        encoder_input: Input features (batch_size x seq_len x 2653)
        vocab: Vocabulary object
        **kwargs: Configuration parameters for the predictor

    Returns:
        Prediction results
    """
    # Set model to evaluation mode
    model.eval()

    # Default configuration tuned for SLT with mode collapse issues
    default_config = {
        'max_length': 100,
        'beam_width': 5,  # As requested
        'temperature': 0.75,  # Between 0.7-0.8 as requested
        'length_penalty_alpha': 0.6,  # Balanced length normalization
        'ngram_blocking_size': 3,  # Increased to 3 (trigrams) for better context
        'frequency_penalty_weight': 0.5, # Changed to scaling factor (0.0-1.0)
        'attention_guided_weight': 2.0,  # Strong feature guidance
        'diversity_penalty_weight': 0.3,  # Encourage diversity
        'bias_words': {'such', 'also', 'easily', 'research', 'should', 'an', 'the', 'a',
                      'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                      'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were'}
    }

    # Update with user-provided config
    default_config.update(kwargs)

    # Make prediction
    results = predict_translation(model, encoder_input, vocab, **default_config)

    return results

def evaluate_prediction_quality(results, expected_translations=None):
    """
    Evaluate the quality of predictions to measure improvement over mode collapse

    Args:
        results: Output from prediction function
        expected_translations: Ground truth translations (optional)

    Returns:
        Dictionary with quality metrics
    """
    metrics = {}

    for i, result in enumerate(results):
        sequence = result['sequence']
        text = result['text']

        # Calculate repetition metrics
        words = text.split()
        unique_words = set(words)
        repetition_rate = 1 - (len(unique_words) / len(words)) if words else 0

        # Check for bias word dominance
        bias_words = {'such', 'also', 'easily', 'research', 'should', 'an', 'the', 'a',
                     'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        bias_word_count = sum(1 for word in words if word.lower() in bias_words)
        bias_word_ratio = bias_word_count / len(words) if words else 0

        metrics[f'sample_{i}'] = {
            'repetition_rate': repetition_rate,
            'bias_word_ratio': bias_word_ratio,
            'sequence_length': len(sequence),
            'unique_words': len(unique_words),
            'total_words': len(words)
        }

    # Overall metrics
    all_repetition_rates = [metrics[key]['repetition_rate'] for key in metrics.keys()]
    all_bias_ratios = [metrics[key]['bias_word_ratio'] for key in metrics.keys()]

    metrics['overall'] = {
        'avg_repetition_rate': sum(all_repetition_rates) / len(all_repetition_rates) if all_repetition_rates else 0,
        'avg_bias_word_ratio': sum(all_bias_ratios) / len(all_bias_ratios) if all_bias_ratios else 0,
        'target_improvement': 'Lower repetition rate and bias word ratio indicate better performance'
    }

    return metrics

def get_model_attention_weights(model, encoder_input, decoder_input):
    """
    Extract attention weights from the model for analysis
    (Requires the model to return attention weights)
    """
    try:
        with torch.no_grad():
            encoder_output = model.encode(encoder_input)
            decoder_output, attention_weights = model.decode(
                decoder_input,
                encoder_output,
                return_attention=True  # Model must support this
            )
        return attention_weights
    except Exception as e:
        logger.warning(f"Could not extract attention weights: {e}")
        return None

def adaptive_temperature_control(loss_history, base_temperature=0.75):
    """
    Adjust temperature based on recent model performance
    Higher temperature when model is overconfident (mode collapse)
    """
    if len(loss_history) < 5:
        return base_temperature

    # Calculate trend in recent losses
    recent_losses = loss_history[-5:]
    if len(set(recent_losses)) == 1:  # No variation - possible mode collapse
        return min(1.0, base_temperature + 0.1)  # Increase temperature

    # If loss is improving consistently
    if recent_losses[-1] < recent_losses[0]:
        return base_temperature  # Maintain current temperature

    # If loss is stagnating, increase randomness
    return min(1.0, base_temperature + 0.05)

# Example usage function
def run_advanced_inference(model, features_batch, vocab, num_samples=5):
    """
    Run advanced inference on a batch of samples
    """
    print("Running advanced inference with contextual constraints...")

    # Sample a few items from the batch
    sample_features = features_batch[:num_samples] if len(features_batch) > num_samples else features_batch

    results = integrate_with_model(
        model,
        sample_features,
        vocab,
        temperature=0.75,
        frequency_penalty_weight=3.0,  # Heavy penalty for bias words
        ngram_blocking_size=2
    )

    # Evaluate quality
    quality_metrics = evaluate_prediction_quality(results)

    print(f"Quality Metrics: {quality_metrics['overall']}")

    # Print sample results
    for i, result in enumerate(results[:3]):  # Show first 3 results
        print(f"\nSample {i+1}:")
        print(f"Text: {result['text']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Sequence: {result['sequence']}")

    return results, quality_metrics