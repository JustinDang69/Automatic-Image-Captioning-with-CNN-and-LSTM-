import numpy as np

def length_normalize_score(raw_score, sequence_length, alpha=0.7):
    """
    Normalizes a sequence score to prevent bias against longer captions in Beam Search.
    
    Formula:
        normalized_score = raw_score / (sequence_length ^ alpha)
        
    Args:
        raw_score (float): The cumulative log-probability of the sequence.
        sequence_length (int): The number of words in the generated sequence.
        alpha (float): The length penalty parameter (typically between 0.6 and 0.8).
        
    Returns:
        float: The length-normalized score.
    """
    safe_length = max(sequence_length, 1)  # Prevent division by zero
    return raw_score / (safe_length ** alpha)

#chaned k=3 to top_avoid to avoid confusion with the k in 
def top_k_sampling(probabilities, top_k=3, seed=None):
    """
    Selects the next word token using Top-k sampling to add controlled variety.
    
    Args:
        probabilities (numpy.ndarray): A 1D or 2D array of probabilities for the next word.
        k (int): The number of top candidates to consider.
        seed (int, optional): Random seed for reproducibility.
        
    Returns:
        int: The index (token ID) of the chosen word as a standard Python integer.
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Flatten array to handle 2D inputs from Keras model.predict() -> shape (1, vocab_size)
    probs_1d = np.asarray(probabilities).flatten()
        
    # Find the indices of the top-k probabilities
    top_k_indices = np.argsort(probs_1d)[-top_k:]
    
    # Get the actual probability values for those indices
    top_k_probs = probs_1d[top_k_indices]

    # Exclude invalid tokens (probabilities <= 0) to prevent math errors during normalization
    valid_mask = top_k_probs > 0
    top_k_indices = top_k_indices[valid_mask]
    top_k_probs = top_k_probs[valid_mask]
    
    # Fallback to pure greedy decoding if all top-k probabilities are invalid
    if len(top_k_probs) == 0:
        return int(np.argmax(probs_1d))

    # Normalize the probabilities to sum to 1
    normalized_probs = top_k_probs / np.sum(top_k_probs)
    
    # Randomly choose one index based on the new weights
    chosen_index = np.random.choice(top_k_indices, p=normalized_probs)
    
    # Cast to standard Python int so it plugs directly into the main decoder without type errors
    return int(chosen_index)