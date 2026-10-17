import numpy as np
import sys
import os
import numpy as np

# Add the src directory to Python's path so it can find your functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from decoding_enhancements import length_normalize_score, top_k_sampling
def test_length_normalization():
    print("--- Testing Length Normalization ---")
    # Scenario: 
    # Caption A is short (raw score -3.2, length 3)
    # Caption B is longer and more descriptive (raw score -5.8, length 7)
    score_a, len_a = -3.2, 3
    score_b, len_b = -5.8, 7
    
    norm_a = length_normalize_score(score_a, len_a, alpha=0.7)
    norm_b = length_normalize_score(score_b, len_b, alpha=0.7)
    
    print(f"Caption A (Length {len_a}): Raw = {score_a}, Normalized = {norm_a:.4f}")
    print(f"Caption B (Length {len_b}): Raw = {score_b}, Normalized = {norm_b:.4f}")
    print(f"Winner after normalization: {'Caption A' if norm_a > norm_b else 'Caption B'}\n")


def test_top_k_sampling():
    print("--- Testing Top-k Sampling ---")
    # Mock vocabulary words and their predicted probabilities
    vocab = ["running", "playing", "sleeping", "flying", "jumping"]
    probabilities = np.array([0.50, 0.30, 0.15, 0.03, 0.02])
    
    k = 3
    seed = 42
    
    print(f"Full Vocabulary Probabilities:")
    for word, prob in zip(vocab, probabilities):
        print(f"  {word}: {prob}")
        
    print(f"\nRunning Top-{k} sampling across 10 trials (with varying seeds)...")
    
    selected_counts = {}
    for i in range(10):
        # We vary the seed slightly each iteration to see sampling variety
        idx = top_k_sampling(probabilities, k=k, seed=seed + i)
        word = vocab[idx]
        selected_counts[word] = selected_counts.get(word, 0) + 1
        
    print(f"Results over 10 trials: {selected_counts}")
    print("Observation: Only the top-k words ('running', 'playing', 'sleeping') are selected.")
    print("Words outside top-k ('flying', 'jumping') have a 0% chance of being chosen.\n")


if __name__ == "__main__":
    test_length_normalization()
    test_top_k_sampling()