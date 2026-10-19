"""Regression and edge-case tests for the decoding innovations (PR #7).

1. With the innovations disabled (top_k=None, alpha=0), greedy_decode and
   beam_search_decode must give exactly the captions of Kevan's approved
   implementation (frozen below from main @ 27c4f77), which produced the
   verified baseline BLEU scores.
2. Top-k sampling and length-normalised beam search edge cases.
3. src.evaluate_bleu CLI: --data-dir, argument validation, output JSON and
   the shared reference-cleaning / BLEU protocol.

Uses the project's real CaptionTokenizer and deterministic mock models; no
trained model, GPU or Flickr8k download is needed.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import src.evaluate_bleu as evaluate_bleu
from src.caption_decoding import (
    CaptionAssets, _as_feature_batch, _join_tokens, _predict_distribution, _special_indices,
    beam_search_decode, greedy_decode,
)
from src.data_preprocessing import CaptionTokenizer
from src.decoding_enhancements import top_k_sampling, validate_top_k
from src.feature_extraction import FEATURE_DIM

WORDS = ["a", "dog", "cat", "runs", "on", "grass", "red", "ball"]
TOK = CaptionTokenizer().fit([f"startseq {' '.join(WORDS)} endseq"])
V = TOK.vocab_size
PAD, UNK = 0, TOK.word_index["<unk>"]
START, EOS = TOK.word_index["startseq"], TOK.word_index["endseq"]
BANNED = {PAD, UNK, START}
VEC = np.zeros(FEATURE_DIM, np.float32)


# --- Kevan's approved baseline, frozen verbatim from main @ 27c4f77 ----------

def reference_greedy(model, feature_vector, tokenizer, max_length, suppress_unk=True):
    image = _as_feature_batch(feature_vector)
    banned, eos = _special_indices(tokenizer, suppress_unk)
    tokens = [tokenizer.word_index["startseq"]]
    generated = []
    for _ in range(max_length - 1):
        probabilities = _predict_distribution(model, image, [tokens], max_length)[0]
        probabilities[list(banned)] = -1
        next_id = int(np.argmax(probabilities))
        if probabilities[next_id] <= 0:
            raise ValueError("No positive-probability valid next word")
        if next_id == eos:
            break
        generated.append(next_id)
        tokens.append(next_id)
    return _join_tokens(generated, tokenizer)


def reference_beam(model, feature_vector, tokenizer, max_length, beam_width=3, suppress_unk=True):
    image = _as_feature_batch(feature_vector)
    banned, eos = _special_indices(tokenizer, suppress_unk)
    start = tokenizer.word_index["startseq"]
    hypotheses = [([start], 0.0, False)]
    for _ in range(max_length - 1):
        active = [h for h in hypotheses if not h[2]]
        if not active:
            break
        vectors = np.repeat(image, len(active), axis=0)
        probabilities = _predict_distribution(model, vectors, [h[0] for h in active], max_length)
        candidates = [h for h in hypotheses if h[2]]
        for hypothesis, row in zip(active, probabilities):
            row = row.copy()
            row[list(banned)] = 0.0
            valid = np.flatnonzero(row > 0)
            if len(valid) == 0:
                candidates.append((hypothesis[0], hypothesis[1], True))
                continue
            k = min(beam_width, len(valid))
            choices = valid[np.argsort(row[valid])[-k:][::-1]]
            for index in choices:
                tokens, log_score, _ = hypothesis
                candidates.append((tokens + [int(index)], log_score + float(np.log(row[index])),
                                   int(index) == eos))
        candidates.sort(key=lambda h: h[1], reverse=True)
        hypotheses = candidates[:beam_width]
    best = max(hypotheses, key=lambda h: h[1])
    return _join_tokens(best[0][1:], tokenizer)


class MockModel:
    """Deterministic next-word distribution for every prefix (a different one per seed).

    ``special_mass`` scales the probability of PAD/<unk>/startseq so that banned
    tokens are often the most likely ones. Probabilities are real-valued (no exact
    ties), like a trained softmax.
    """

    def __init__(self, seed, sharpness=1.0, special_mass=1.0):
        self.seed, self.sharpness, self.special_mass = seed, sharpness, special_mass
        self.calls = 0

    def row(self, prefix):
        h = int(hashlib.md5(f"{self.seed}|{prefix}".encode()).hexdigest()[:8], 16)
        r = np.random.default_rng(h).dirichlet(np.full(V, self.sharpness))
        r[list(BANNED)] *= self.special_mass
        return r / r.sum()

    def predict_on_batch(self, inputs):
        self.calls += 1
        assert inputs["image_features"].shape[1] == FEATURE_DIM
        return np.stack([self.row(tuple(int(t) for t in p if t != 0)) for p in inputs["caption_input"]])


def regression_cases(n=300):
    rng = np.random.default_rng(2026)
    for i in range(n):
        model = MockModel(i, sharpness=float(rng.choice([0.2, 0.5, 1.0, 3.0])),
                          special_mass=float(rng.choice([0.0, 1.0, 5.0])))
        yield model, int(rng.integers(2, 12)), int(rng.integers(1, 7))


# --- 1. Baselines are preserved -------------------------------------------------

def test_greedy_with_top_k_disabled_matches_approved_baseline():
    for model, max_length, _ in regression_cases():
        assert greedy_decode(model, VEC, TOK, max_length) == reference_greedy(model, VEC, TOK, max_length)


def test_beam_with_alpha_zero_matches_approved_baseline():
    for model, max_length, width in regression_cases():
        expected = reference_beam(model, VEC, TOK, max_length, beam_width=width)
        assert beam_search_decode(model, VEC, TOK, max_length, beam_width=width, alpha=0.0) == expected
        assert beam_search_decode(model, VEC, TOK, max_length, beam_width=width) == expected  # default


def test_beam_width_one_and_top_k_one_match_greedy():
    for seed in range(100):
        model = MockModel(seed, sharpness=0.5)
        greedy = greedy_decode(model, VEC, TOK, 10)
        assert beam_search_decode(model, VEC, TOK, 10, beam_width=1) == greedy
        assert greedy_decode(model, VEC, TOK, 10, top_k=1, rng=np.random.default_rng(0)) == greedy


# --- 2a. Top-k sampling -----------------------------------------------------------

@pytest.mark.parametrize("bad", [0, -1, -3, 1.5, "3", None, True])
def test_invalid_top_k_is_rejected(bad):
    with pytest.raises(ValueError, match="top_k"):
        validate_top_k(bad)
    if bad is not None:                               # None means "greedy" in greedy_decode
        model = MockModel(0)
        with pytest.raises(ValueError, match="top_k"):
            greedy_decode(model, VEC, TOK, 10, top_k=bad, rng=np.random.default_rng(0))
        assert model.calls == 0                       # rejected before any model call


def test_top_k_never_samples_banned_tokens_even_when_they_dominate():
    row = MockModel(7, sharpness=0.3, special_mass=50.0).row((START,))
    assert row[list(BANNED)].sum() > 0.5 and int(np.argmax(row)) in BANNED   # banned tokens dominate
    rng = np.random.default_rng(42)
    picks = {top_k_sampling(row, top_k=3, rng=rng, banned_indices=BANNED) for _ in range(2000)}
    allowed = [i for i in np.argsort(row)[::-1] if i not in BANNED][:3]
    assert picks <= set(allowed) and not picks & BANNED
    assert len(picks) == 3                            # samples, rather than always the max


def test_top_k_raises_when_all_valid_probabilities_are_zero():
    row = np.zeros(V)
    row[list(BANNED)] = 1 / 3                         # only banned tokens have probability
    with pytest.raises(ValueError, match="No positive-probability"):
        top_k_sampling(row, top_k=3, rng=np.random.default_rng(0), banned_indices=BANNED)


def test_top_k_is_reproducible_without_repeating_the_same_draw():
    def run(seed):
        rng = np.random.default_rng(seed)             # one generator for the whole "test set"
        return [greedy_decode(MockModel(s), VEC, TOK, 12, top_k=3, rng=rng) for s in range(20)]
    assert run(42) == run(42)
    assert run(42) != run(7)
    rng = np.random.default_rng(42)
    draws = [top_k_sampling(np.full(V, 1 / V), top_k=V, rng=rng) for _ in range(20)]
    assert len(set(draws)) > 1                        # not reseeded at every token


def test_top_k_respects_max_length_and_stops_at_endseq():
    for seed in range(50):
        caption = greedy_decode(MockModel(seed), VEC, TOK, 6, top_k=3, rng=np.random.default_rng(seed))
        assert len(caption.split()) <= 5              # max_length - 1 words after startseq
        assert not {"startseq", "endseq", "<unk>"} & set(caption.split())


# --- 2b. Length-normalised beam search -------------------------------------------

class EarlyStopModel:
    """startseq -> endseq with probability p_end, otherwise a fixed 8-word caption (p=0.75 per word)."""

    def __init__(self, p_end):
        self.p_end = p_end

    def predict_on_batch(self, inputs):
        rows = []
        for prefix in inputs["caption_input"]:
            words = [int(t) for t in prefix if t != 0]
            r = np.full(V, 1e-6)
            if words == [START]:
                r[EOS], r[TOK.word_index["a"]] = self.p_end, 1 - self.p_end
            elif len(words) < 9:
                r[TOK.word_index[WORDS[len(words) % len(WORDS)]]], r[EOS] = 0.75, 0.25
            else:
                r[EOS] = 1.0
            rows.append(r / r.sum())
        return np.stack(rows)


@pytest.mark.parametrize("p_end", [0.2, 0.4])
def test_length_norm_avoids_the_empty_caption_standard_beam_prefers(p_end):
    model = EarlyStopModel(p_end)
    assert beam_search_decode(model, VEC, TOK, 39, beam_width=3) == ""          # baseline bias
    long_caption = beam_search_decode(model, VEC, TOK, 39, beam_width=3, alpha=0.7)
    assert len(long_caption.split()) == 8


def test_length_norm_ranks_by_score_over_tokens_including_startseq():
    # [startseq, a, endseq] (3 tokens) vs [startseq, a, dog, cat, endseq] (5 tokens)
    from src.decoding_enhancements import length_normalize_score
    short, long_ = -2.0, -2.5
    assert short > long_                              # raw log-prob would pick the short one
    # -2.0 / 3**0.7 = -0.927  <  -2.5 / 5**0.7 = -0.810  -> length norm picks the long one
    assert length_normalize_score(short, 3, 0.7) < length_normalize_score(long_, 5, 0.7)


def test_negative_alpha_is_rejected():
    with pytest.raises(ValueError, match="alpha"):
        beam_search_decode(MockModel(0), VEC, TOK, 10, alpha=-0.5)


def test_decoders_accept_the_saved_project_tokenizer():
    path = Path(__file__).resolve().parent.parent / "data" / "processed" / "tokenizer.json"
    if not path.is_file():
        pytest.skip("data/processed/tokenizer.json not created (run notebook 01)")
    tok = CaptionTokenizer.load(path)
    assert tok.vocab_size == 7702

    class FullVocabModel:
        def predict_on_batch(self, inputs):
            assert inputs["caption_input"].shape[1] == 39
            return np.random.default_rng(0).dirichlet(np.ones(tok.vocab_size), len(inputs["caption_input"]))

    for caption in (greedy_decode(FullVocabModel(), VEC, tok, 39, top_k=3, rng=np.random.default_rng(42)),
                    beam_search_decode(FullVocabModel(), VEC, tok, 39, beam_width=3, alpha=0.7)):
        assert all(w in tok.word_index for w in caption.split())
        assert not {"startseq", "endseq", "<unk>"} & set(caption.split())


# --- 3. evaluate_bleu CLI ---------------------------------------------------------

class PerfectModel:
    """Always continues 'a dog runs on grass' then endseq (p=0.97 per step)."""

    caption = ["a", "dog", "runs", "on", "grass"]

    def predict_on_batch(self, inputs):
        rows = []
        for prefix in inputs["caption_input"]:
            n = len([t for t in prefix if t != 0]) - 1           # words generated so far
            nxt = TOK.word_index[self.caption[n]] if n < len(self.caption) else EOS
            r = np.full(V, 0.03 / (V - 1))
            r[nxt] = 0.97
            rows.append(r)
        return np.stack(rows)


@pytest.fixture
def fake_assets(monkeypatch, tmp_path):
    """Two test images whose raw references need cleaning (case, punctuation, quotes)."""
    refs = {i: ["A dog runs on grass .", "A DOG runs on grass", "\"a dog , runs on grass\"",
                "a dog runs on grass!", "A dog runs on grass ."] for i in ("x.jpg", "y.jpg")}
    assets = CaptionAssets(model=PerfectModel(), tokenizer=TOK, max_length=39,
                           splits={"train": [], "val": [], "test": ["x.jpg", "y.jpg"]},
                           features=np.zeros((2, FEATURE_DIM), np.float32), row_of={"x.jpg": 0, "y.jpg": 1},
                           images_dir=tmp_path, references=refs)
    seen = {}

    def fake_load_assets(model_dir, features_dir, data_dir=None):
        seen.update(model_dir=model_dir, features_dir=features_dir, data_dir=data_dir)
        return assets
    monkeypatch.setattr(evaluate_bleu, "load_assets", fake_load_assets)
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "caption_model.keras").write_bytes(b"")
    (tmp_path / "run" / "training_config.json").write_text(json.dumps({"best_epoch": 5, "best_val_loss": 3.3437}))
    return seen, tmp_path


def test_cli_passes_data_dir_and_writes_complete_json(fake_assets):
    seen, tmp = fake_assets
    out = tmp / "decoding_results" / "innovations.json"
    evaluate_bleu.main(["--model-dir", str(tmp / "run"), "--features-dir", "F", "--data-dir", "D",
                        "--top-k", "3", "--seed", "42", "--beam-width", "3", "--alpha", "0.7",
                        "--output-json", str(out)])
    assert seen == {"model_dir": str(tmp / "run"), "features_dir": "F", "data_dir": "D"}
    result = json.loads(out.read_text())
    assert result["image_count"] == 2 and result["full_test_set"] is True
    assert result["model_file"].endswith("caption_model.keras") and result["best_epoch"] == 5
    assert (result["top_k"], result["seed"], result["beam_width"], result["alpha"]) == (3, 42, 3, 0.7)
    assert result["max_length"] == 39 and result["data_dir"] == "D"
    assert set(result["scores"]) == {"top_k", "length_norm"}
    for method in ("top_k", "length_norm"):
        assert set(result["scores"][method]) == {"BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"}
        assert len(result["captions"][method]) == 2 and result["empty_captions"][method] == 0
    # length-norm beam finds the perfect caption; cleaned references -> BLEU = 1.0
    assert result["captions"]["length_norm"]["x.jpg"] == "a dog runs on grass"
    assert all(v == pytest.approx(1.0) for v in result["scores"]["length_norm"].values())


def test_cli_uses_cleaned_references_like_the_baseline(fake_assets):
    seen, tmp = fake_assets
    out = tmp / "r.json"
    evaluate_bleu.main(["--model-dir", str(tmp / "run"), "--top-k", "1", "--methods", "top_k",
                        "--output-json", str(out)])
    result = json.loads(out.read_text())
    assert set(result["scores"]) == {"top_k"}
    assert result["scores"]["top_k"]["BLEU-4"] == pytest.approx(1.0)   # punctuation/case/quotes removed


@pytest.mark.parametrize("args", [["--top-k", "0"], ["--top-k", "-2"], ["--beam-width", "0"], ["--alpha", "-1"]])
def test_cli_rejects_invalid_parameters(args, fake_assets):
    with pytest.raises(SystemExit):
        evaluate_bleu.main(args)


def test_bleu3_weights_are_exactly_one_third():
    import inspect
    assert "(1/3, 1/3, 1/3, 0)" in inspect.getsource(evaluate_bleu.evaluate_model)
