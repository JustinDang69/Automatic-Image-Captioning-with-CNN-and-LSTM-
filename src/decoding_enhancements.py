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
def top_k_sampling(probabilities, top_k=3, rng=None, banned_indices=None):
    if rng is None:
        rng = np.random.default_rng()
        
    probs_1d = np.asarray(probabilities).flatten().copy()
    
    # Zero out banned tokens (PAD, <unk>, startseq)
    if banned_indices:
        probs_1d[list(banned_indices)] = -1.0
        
    top_k_indices = np.argsort(probs_1d)[-top_k:]
    top_k_probs = probs_1d[top_k_indices]

    valid_mask = top_k_probs > 0
    top_k_indices = top_k_indices[valid_mask]
    top_k_probs = top_k_probs[valid_mask]
    
    if len(top_k_probs) == 0:
        return int(np.argmax(probs_1d))

    normalized_probs = top_k_probs / np.sum(top_k_probs)
    return int(rng.choice(top_k_indices, p=normalized_probs))