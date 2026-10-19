"""Generate Flickr8k captions with greedy and standard beam-search decoding.

Uses the *same* trained CNN-feature + LSTM next-word model for both methods:
    saved VGG16 fc2 (4096) + padded prefix -> vocabulary probabilities.

Requires training artifacts from src.train_lstm (especially caption_model.keras,
tokenizer.json, metadata.json, splits.json), plus saved VGG16 features.
No Flickr8k data or trained model is bundled with this source code.

Run from repository root:
    python -m src.caption_decoding --model-dir models/lstm --beam-width 3
    python -m src.caption_decoding --model-dir models/lstm --beam-width 5 --eval-test

One configurable beam width is used per run. Output files are saved in a
beam-width-specific folder, so changing k does not overwrite prior experiments.
"""
from __future__ import annotations

import argparse
import json
import random
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from keras.utils import pad_sequences

from src.data_preprocessing import (
    CaptionTokenizer, clean_caption, get_dataset_paths, load_captions,
)
from src.feature_extraction import DEFAULT_FEATURES_DIR, FEATURE_DIM, load_features
from src.decoding_enhancements import length_normalize_score, top_k_sampling, validate_top_k


def _special_indices(tokenizer, suppress_unk=True):
    """Never generate PAD or startseq; optionally suppress unknown words."""
    banned = {0}
    for word in (("startseq", "<unk>") if suppress_unk else ("startseq",)):
        if word in tokenizer.word_index:
            banned.add(tokenizer.word_index[word])
    if "endseq" not in tokenizer.word_index or "startseq" not in tokenizer.word_index:
        raise ValueError("Saved tokenizer must contain startseq and endseq")
    return banned, tokenizer.word_index["endseq"]


def _predict_distribution(model, vectors, prefixes, max_length):
    """One batched model call, compatible with the saved two-input Keras model."""
    features = np.asarray(vectors, dtype=np.float32)
    words = pad_sequences(prefixes, maxlen=max_length, padding="pre",
                          truncating="pre", dtype="int32")
    probs = np.asarray(model.predict_on_batch({
        "image_features": features,
        "caption_input": words,
    }), dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] != len(prefixes):
        raise ValueError(f"Unexpected model predictions shape: {probs.shape}")
    if not np.isfinite(probs).all() or (probs < 0).any():
        raise ValueError("Invalid next-word probabilities from the model")
    return probs


def _as_feature_batch(feature_vector):
    """Accept a single 4096-d feature vector, not an image or a feature matrix."""
    feature = np.asarray(feature_vector, dtype=np.float32)
    if feature.shape != (FEATURE_DIM,):
        raise ValueError(f"Expected VGG16 fc2 feature shape ({FEATURE_DIM},), got {feature.shape}")
    return feature.reshape(1, FEATURE_DIM)


def _join_tokens(indices, tokenizer):
    """Readable caption without special markers."""
    words = [tokenizer.index_word.get(int(i), "<unk>") for i in indices]
    return " ".join(w for w in words if w not in ("startseq", "endseq", "<unk>"))


def greedy_decode(model, feature_vector, tokenizer, max_length, suppress_unk=True,
                  top_k=None, rng=None):
    """Choose the most likely valid next word at every step.

    Greedy is fast but makes a local decision that cannot be undone.

    Innovation 2 - Top-k sampling (Abdul): if ``top_k`` is set, the next word is
    instead *sampled* from the ``top_k`` most likely valid words, with their
    probabilities renormalised to sum to 1 (``decoding_enhancements.top_k_sampling``).
    PAD, startseq and <unk> are never sampled. ``top_k=None`` (default) is the
    original greedy baseline, unchanged.

    Args:
        top_k: None for greedy decoding, or a positive integer k (k=1 is greedy).
        rng: ``np.random.Generator`` used for Top-k sampling. Create it ONCE per
            evaluation run, e.g. ``np.random.default_rng(42)``, and pass the same
            generator for every image: results are then reproducible and each word
            gets a fresh random draw. If None, an unseeded generator is created for
            this caption (not reproducible).
    """
    if max_length < 2:
        raise ValueError("max_length must be >= 2")
    if top_k is not None:
        validate_top_k(top_k)
        if rng is None:
            rng = np.random.default_rng()
    image = _as_feature_batch(feature_vector)
    banned, eos = _special_indices(tokenizer, suppress_unk)
    tokens = [tokenizer.word_index["startseq"]]
    generated = []

    for _ in range(max_length - 1):
        probabilities = _predict_distribution(model, image, [tokens], max_length)[0]

        if top_k is not None:
            # Banned tokens are excluded inside top_k_sampling, which raises
            # ValueError if no valid word has a positive probability.
            next_id = top_k_sampling(probabilities, top_k=top_k, rng=rng, banned_indices=banned)
        else:
            probabilities[list(banned)] = -1
            next_id = int(np.argmax(probabilities))

        if probabilities[next_id] <= 0 and top_k is None:
            raise ValueError("No positive-probability valid next word")
        if next_id == eos:
            break

        generated.append(next_id)
        tokens.append(next_id)

    return _join_tokens(generated, tokenizer)


def beam_search_decode(model, feature_vector, tokenizer, max_length,
                       beam_width=3, suppress_unk=True, alpha=0.0):
    """Standard beam search ranked by cumulative log probability (no length penalty).

    At each step, expand up to beam_width promising unfinished hypotheses.
    Finished hypotheses are retained without expanding them. The highest-scoring
    finished or truncated sequence is returned once all beams stop or max length
    is reached. Model calls are batched across active beams.

    Innovation 1 - Length-normalised beam search (Abdul): if ``alpha > 0``,
    hypotheses are ranked - both when pruning to ``beam_width`` and when choosing
    the final caption - by ``log_score / length ** alpha``
    (``decoding_enhancements.length_normalize_score``). ``length`` is the number of
    tokens in the hypothesis, *including* startseq (and endseq once finished).
    Summed log probabilities only get more negative as words are added, so
    standard beam search prefers short captions; dividing by length**alpha
    reduces that bias. ``alpha=0.0`` (default) is the original standard beam
    search, unchanged.

    Args:
        alpha: length-penalty strength, >= 0 (e.g. 0.7). 0 = standard beam search.
    """
    if not isinstance(beam_width, int) or beam_width < 1:
        raise ValueError("beam_width must be a positive integer")
    if max_length < 2:
        raise ValueError("max_length must be >= 2")
    if alpha < 0:
        raise ValueError("alpha must be >= 0 (0 = standard beam search)")

    image = _as_feature_batch(feature_vector)
    banned, eos = _special_indices(tokenizer, suppress_unk)
    start = tokenizer.word_index["startseq"]
    # (token IDs including startseq, summed log probability, is_finished)
    hypotheses = [([start], 0.0, False)]

    for _ in range(max_length - 1):
        active = [h for h in hypotheses if not h[2]]
        if not active:
            break
        vectors = np.repeat(image, len(active), axis=0)
        probabilities = _predict_distribution(
            model, vectors, [h[0] for h in active], max_length
        )
        candidates = [h for h in hypotheses if h[2]]

        for hypothesis, row in zip(active, probabilities):
            row = row.copy()
            row[list(banned)] = 0.0
            # Rank possible continuations using *actual* probabilities.
            valid = np.flatnonzero(row > 0)
            if len(valid) == 0:
                candidates.append((hypothesis[0], hypothesis[1], True))
                continue

            k_limit = min(beam_width, len(valid))
            choices = valid[np.argsort(row[valid])[-k_limit:][::-1]]

            for index in choices:
                tokens, log_score, _ = hypothesis
                candidates.append((
                    tokens + [int(index)],
                    log_score + float(np.log(row[index])),
                    int(index) == eos,
                ))

        if alpha > 0.0:
            candidates.sort(key=lambda h: length_normalize_score(h[1], len(h[0]), alpha), reverse=True)
        else:
            candidates.sort(key=lambda h: h[1], reverse=True)

        hypotheses = candidates[:beam_width]

    # `hypotheses` is sorted by the ranking score above, so [0] is the best one.
    # With alpha=0 this is identical to the original max(..., key=log score).
    best = hypotheses[0]
    return _join_tokens(best[0][1:], tokenizer)


def _reference_tokens(caption):
    """Match the project's caption cleaning without start/end markers."""
    words = clean_caption(caption).split()
    return words[1:-1]


@dataclass
class CaptionAssets:
    model: Any
    tokenizer: CaptionTokenizer
    max_length: int
    splits: dict
    features: np.ndarray
    row_of: dict
    images_dir: Path
    references: dict


def load_assets(model_dir, features_dir=DEFAULT_FEATURES_DIR, data_dir=None):
    """Load the *trained* best decoder, tokenizer, exact splits and Flickr8k data."""
    import keras

    model_dir = Path(model_dir)
    model_file = model_dir / "caption_model.keras"
    if not model_file.exists():
        model_file = model_dir / "best_caption_model.keras"
    if not model_file.exists():
        raise FileNotFoundError(
            f"No trained model in {model_dir}. Copy caption_model.keras "
            "from your Colab/Google Drive training run."
        )
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    splits = json.loads((model_dir / "splits.json").read_text(encoding="utf-8"))
    tokenizer = CaptionTokenizer.load(model_dir / "tokenizer.json")
    max_length = int(metadata["max_length"])
    model = keras.models.load_model(model_file, compile=False)
    if model.output_shape[-1] != tokenizer.vocab_size:
        raise ValueError("Model output vocabulary size does not match tokenizer.json")
    features, row_of = load_features(features_dir, require_complete=True, mmap=True)
    images_dir, caption_file = get_dataset_paths(data_dir)
    references = load_captions(caption_file)
    return CaptionAssets(model, tokenizer, max_length, splits,
                         features, row_of, images_dir, references)


def decode_image(assets, image_id, beam_width=3):
    """Generate exactly one greedy and one beam-search caption for an image.

    `beam_width` controls how many *internal candidate sequences* beam search
    retains. There is only one beam-search output per image per run.
    """
    if not isinstance(beam_width, int) or beam_width < 1:
        raise ValueError("beam_width must be a positive integer")
    if image_id not in assets.row_of:
        raise KeyError(f"Missing VGG16 features for {image_id}")
    if image_id not in assets.references:
        raise KeyError(f"Missing Flickr8k reference captions for {image_id}")
    vector = assets.features[assets.row_of[image_id]]

    t0 = time.perf_counter()
    greedy = greedy_decode(assets.model, vector, assets.tokenizer, assets.max_length)
    greedy_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    beam = beam_search_decode(
        assets.model, vector, assets.tokenizer, assets.max_length,
        beam_width=beam_width,
    )
    beam_time = time.perf_counter() - t0

    return {
        "image_id": image_id,
        "references": [" ".join(_reference_tokens(c)) for c in assets.references[image_id]],
        "greedy": greedy,
        "beam": beam,
        "beam_width": beam_width,
        "seconds": {
            "greedy": round(greedy_time, 3),
            "beam": round(beam_time, 3),
        },
    }


def example_captions(assets, num_examples=6, seed=42, beam_width=3,
                     split="test"):
    """Reproducibly sample held-out images; run greedy + ONE beam width."""
    if split not in ("test", "val"):
        raise ValueError("Use validation or held-out test images for examples")
    if num_examples <= 0:
        raise ValueError("num_examples must be positive")
    if not isinstance(beam_width, int) or beam_width < 1:
        raise ValueError("beam_width must be a positive integer")
    ids = assets.splits[split]
    if not ids:
        raise ValueError(f"No images in {split} split")
    chosen = random.Random(seed).sample(ids, min(num_examples, len(ids)))
    return [decode_image(assets, image_id, beam_width=beam_width)
            for image_id in chosen]


def save_example_figure(assets, examples, output_file):
    """Figure with the actual image, reference captions and each decoding result."""
    import matplotlib.pyplot as plt
    from PIL import Image

    n = len(examples)
    fig, axes = plt.subplots(n, 2, figsize=(15, max(4 * n, 4)),
                             squeeze=False, gridspec_kw={"width_ratios": [1, 1.4]})
    for ax_pair, item in zip(axes, examples):
        image_ax, text_ax = ax_pair
        with Image.open(assets.images_dir / item["image_id"]) as im:
            image_ax.imshow(im.convert("RGB"))
        image_ax.set_title(item["image_id"], fontsize=10)
        image_ax.axis("off")
        lines = ["Human reference captions:"]
        lines += [textwrap.fill(f"  {i}. {ref}", width=67, subsequent_indent="     ")
                  for i, ref in enumerate(item["references"], start=1)]
        lines += [
            "",
            "Greedy: " + item["greedy"],
            f"Beam (k={item['beam_width']}): " + item["beam"],
        ]
        text_ax.text(0.01, 0.97, "\n".join(
            textwrap.fill(line, width=72, subsequent_indent="    ") if len(line) > 72 else line
            for line in lines
        ), va="top", ha="left", fontsize=9, transform=text_ax.transAxes,
                     linespacing=1.4)
        text_ax.axis("off")
    fig.tight_layout(h_pad=3.5)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=140, bbox_inches="tight")
    plt.close(fig)

def evaluate_bleu(assets, beam_width=3, max_images=None):
    """Compare greedy against a SINGLE beam width using corpus BLEU-1..4.

    Final evaluation should use all held-out test images (max_images=None).
    Restricted image counts are exploratory, not final performance results.
    """
    from nltk.translate.bleu_score import corpus_bleu

    if not isinstance(beam_width, int) or beam_width < 1:
        raise ValueError("beam_width must be a positive integer")
    ids = assets.splits["test"]
    if max_images is not None:
        if max_images <= 0:
            raise ValueError("max_images must be positive")
        ids = ids[:max_images]
    if not ids:
        raise ValueError("No held-out test images available")

    references = []
    hypotheses = {"greedy": [], "beam": []}
    elapsed = {"greedy": [], "beam": []}
    for i, image_id in enumerate(ids, 1):
        item = decode_image(assets, image_id, beam_width=beam_width)
        references.append([_reference_tokens(c) for c in assets.references[image_id]])
        for method in hypotheses:
            hypotheses[method].append(item[method].split())
            elapsed[method].append(item["seconds"][method])
        if i % 100 == 0:
            print(f"Decoded {i}/{len(ids)} test images", flush=True)

    weights = {
        "BLEU-1": (1., 0., 0., 0.),
        "BLEU-2": (.5, .5, 0., 0.),
        "BLEU-3": (1 / 3, 1 / 3, 1 / 3, 0.),
        "BLEU-4": (.25, .25, .25, .25),
    }
    scores = {}
    for method, captions in hypotheses.items():
        scores[method] = {
            name: float(corpus_bleu(references, captions, weights=weight))
            for name, weight in weights.items()
        }
        scores[method]["avg_seconds_per_caption"] = float(np.mean(elapsed[method]))
    return {
        "split": "test",
        "beam_width": beam_width,
        "images_evaluated": len(ids),
        "full_test_set": len(ids) == len(assets.splits["test"]),
        "scores": scores,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Flickr8k greedy and beam-search decoding")
    parser.add_argument("--model-dir", default="models/lstm",
                        help="Folder containing trained caption_model.keras and tokenizer/splits")
    parser.add_argument("--features-dir", default=str(DEFAULT_FEATURES_DIR))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default="models/lstm/decoding_results")
    parser.add_argument("--num-examples", type=int, default=6)
    parser.add_argument("--beam-width", type=int, default=3,
                        help="One beam width k per run, e.g. 3 or 5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-test", action="store_true",
                        help="Calculate corpus BLEU-1 to BLEU-4 on the test split")
    parser.add_argument("--eval-limit", type=int, default=None,
                        help="Exploratory evaluation on only the first N test images")
    args = parser.parse_args(argv)
    if args.beam_width < 1:
        parser.error("--beam-width must be >= 1")

    assets = load_assets(args.model_dir, args.features_dir, args.data_dir)
    output = Path(args.output_dir) / f"beam_k{args.beam_width}"
    output.mkdir(parents=True, exist_ok=True)

    examples = example_captions(assets, args.num_examples, args.seed,
                                beam_width=args.beam_width)
    for item in examples:
        print(f"\\n{item['image_id']}")
        print("Reference:", item["references"][0])
        print("Greedy:   ", item["greedy"])
        print(f"Beam k={args.beam_width}:", item["beam"])

    (output / "caption_examples.json").write_text(
        json.dumps(examples, indent=2), encoding="utf-8"
    )
    save_example_figure(assets, examples, output / "caption_examples.png")
    print(f"\\nSaved examples to {output.resolve()}")

    if args.eval_test:
        result = evaluate_bleu(assets, beam_width=args.beam_width,
                               max_images=args.eval_limit)
        (output / "bleu_comparison.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print("Corpus BLEU results:", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
