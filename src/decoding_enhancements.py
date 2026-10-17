import numpy as np

def length_normalize_score(raw_score, sequence_length, alpha=0.7):
    """
    Normalizes a sequence score to prevent bias against longer captions in Beam Search.
    
    Args:
        raw_score (float): The cumulative log-probability of the sequence.
        sequence_length (int): The number of words in the generated sequence.
        alpha (float): The length penalty parameter (typically between 0.6 and 0.8).
        
    Returns:
        float: The length-normalized score.
    """
    safe_length = max(sequence_length, 1)  # Prevent division by zero
    return raw_score / (safe_length ** alpha)


def top_k_sampling(probabilities, k=3, seed=None):
    """
    Selects the next word token using Top-k sampling to add controlled variety.
    
    Args:
        probabilities (numpy.ndarray): A 1D array of probabilities for the next word.
        k (int): The number of top candidates to consider.
        seed (int, optional): Random seed for reproducibility.
        
    Returns:
        int: The index (token ID) of the chosen word.
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Step 1: Find the indices of the top-k probabilities
    top_k_indices = np.argsort(probabilities)[-k:]

    
    # Step 2: Get the actual probability values for those indices
    top_k_probs = probabilities[top_k_indices]

    # Step 3: Normalize the probabilities to sum to 1
    normalized_probs = top_k_probs / np.sum(top_k_probs)
    
    # Step 4: Randomly choose one index based on the new weights
    chosen_index = np.random.choice(top_k_indices, p=normalized_probs)
    
    return chosen_index

