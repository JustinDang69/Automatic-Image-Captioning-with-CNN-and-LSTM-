"""Decoding innovations (Abdul): length-normalised beam scores and Top-k sampling.

Both operate AFTER the softmax at inference time; the trained model is unchanged.
They are used by ``src.caption_decoding.beam_search_decode(alpha=...)`` and
``src.caption_decoding.greedy_decode(top_k=..., rng=...)``.
"""
import numpy as np

# Purpose: Adjusts a cumulative log-probability by sequence length raised to alpha.
# Implementation: The adjustment reduces the short-sequence preference that arises when negative log-probabilities are summed.
def length_normalize_score(raw_score, sequence_length, alpha=0.7):
    """
    Normalizes a sequence score to prevent bias against longer captions in Beam Search.

    Formula:
        normalized_score = raw_score / (sequence_length ^ alpha)

    Args:
        raw_score (float): The cumulative log-probability of the sequence.
        sequence_length (int): The number of tokens in the hypothesis. Beam search
            passes len(tokens), which includes startseq (and endseq once finished).
        alpha (float): The length penalty parameter (typically between 0.6 and 0.8).

    Returns:
        float: The length-normalized score.
    """
    safe_length = max(sequence_length, 1)  # Prevent division by zero
    return raw_score / (safe_length ** alpha)


# Purpose: validate top k; see the function docstring for parameters, outputs and validation behavior.
def validate_top_k(top_k):
    """Raise ValueError unless ``top_k`` is a positive integer (k=1 behaves like greedy)."""
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k < 1:
        raise ValueError(f"top_k must be a positive integer (>= 1), got {top_k!r}")


# Parameter renamed from k to top_k to avoid confusion with the beam width k.
# Purpose: Samples from a restricted set of likely next-token candidates after excluding prohibited token IDs.
# Implementation: Probabilities are renormalized within the allowed top-k set before drawing a token.
def top_k_sampling(probabilities, top_k=3, rng=None, banned_indices=None):
    """Sample the next token from the ``top_k`` most likely valid tokens.

    Args:
        probabilities: next-word probabilities, shape (vocab_size,) or (1, vocab_size).
        top_k: positive integer; how many of the most likely valid tokens to consider.
        rng: ``np.random.Generator``. Reuse one generator for a whole evaluation run
            (e.g. ``np.random.default_rng(42)``) so results are reproducible without
            reseeding at every token. None = a new unseeded generator.
        banned_indices: token IDs that must never be chosen (PAD, startseq, <unk>).

    Returns:
        int: the sampled token ID.

    Raises:
        ValueError: if ``top_k < 1``, or if no valid (non-banned) token has a
            positive probability - the same condition makes greedy decoding fail.
    """
    validate_top_k(top_k)
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
        # Every valid token has probability 0: there is nothing sensible to sample
        # (argmax would return an arbitrary zero-probability word).
        raise ValueError("No positive-probability valid next word for Top-k sampling")

    normalized_probs = top_k_probs / np.sum(top_k_probs)
    return int(rng.choice(top_k_indices, p=normalized_probs))
