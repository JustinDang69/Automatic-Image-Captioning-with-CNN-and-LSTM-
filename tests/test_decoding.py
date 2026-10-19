import sys
import os
import numpy as np

# Add the src directory to Python's path so it can find your functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from decoding_enhancements import length_normalize_score, top_k_sampling

def test_length_normalization():
    score_a, len_a = -3.2, 3
    score_b, len_b = -5.8, 7
    alpha = 0.7
    
    norm_a = length_normalize_score(score_a, len_a, alpha=alpha)
    norm_b = length_normalize_score(score_b, len_b, alpha=alpha)
    
    assert isinstance(norm_a, float), "Normalized score must be a standard float."
    assert norm_a < 0, "Log-probability normalized score must remain negative."
    
    # Calculate expected values dynamically to avoid hardware-level floating-point precision issues
    expected_a = score_a / (len_a ** alpha)
    expected_b = score_b / (len_b ** alpha)
    
    assert round(norm_a, 4) == round(expected_a, 4), f"Expected {expected_a:.4f}, got {norm_a:.4f}"
    assert round(norm_b, 4) == round(expected_b, 4), f"Expected {expected_b:.4f}, got {norm_b:.4f}"


def test_top_k_sampling():
    # 1. Test 1D array extraction
    probs_1d = np.array([0.50, 0.30, 0.15, 0.03, 0.02])
    rng = np.random.default_rng(42)
    idx_1d = top_k_sampling(probs_1d, top_k=3, rng=rng)
    assert idx_1d in [0, 1, 2], "Chosen index must be restricted to the top 3 probabilities."
    assert isinstance(idx_1d, int), "Return type must be a standard Python integer."
    
    # 2. Test 2D array handling (Simulating Keras predict shape)
    probs_2d = np.array([[0.50, 0.30, 0.15, 0.03, 0.02]])
    rng = np.random.default_rng(42)
    idx_2d = top_k_sampling(probs_2d, top_k=3, rng=rng)
    assert idx_2d == idx_1d, "1D and 2D inputs must yield identical sampling results given the same seed."
    
    # 3. Test invalid token exclusion (masking zero probabilities)
    probs_invalid = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    idx_invalid = top_k_sampling(probs_invalid, top_k=3)
    assert idx_invalid == 3, "Algorithm must exclude zero probabilities and select the only valid token."

if __name__ == "__main__":
    test_length_normalization()
    test_top_k_sampling()
    print("All decoding enhancement assertions passed successfully.")