"""Generate Flickr8k captions with greedy and standard beam-search decoding.

Uses the *same* trained CNN-feature + LSTM next-word model for both methods:
    saved VGG16 fc2 (4096) + padded prefix -> vocabulary probabilities.

Requires training artifacts from src.train_lstm (especially caption_model.keras,
tokenizer.json, metadata.json, splits.json), plus saved VGG16 features.
No Flickr8k data or trained model is bundled with this source code.

Run from repository root:
    python -m src.caption_decoding --model-dir models/lstm --num-examples 6
    python -m src.caption_decoding --model-dir models/lstm --eval-test

Output: reference/greedy/beam captions in caption_examples.json, a PNG
comparison figure.
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


def greedy_decode(model, feature_vector, tokenizer, max_length, suppress_unk=True):
    """Choose the most likely valid next word at every step.

    Greedy is fast but makes a local decision that cannot be undone.
    """
    if max_length < 2:
        raise ValueError("max_length must be >= 2")
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


def beam_search_decode(model, feature_vector, tokenizer, max_length,
                       beam_width=3, suppress_unk=True):
    """Standard beam search ranked by cumulative log probability (no length penalty).

    At each step, expand up to beam_width promising unfinished hypotheses.
    Finished hypotheses are retained without expanding them. The highest-scoring
    finished or truncated sequence is returned once all beams stop or max length
    is reached. Model calls are batched across active beams.
    """
    if not isinstance(beam_width, int) or beam_width < 1:
        raise ValueError("beam_width must be a positive integer")
    if max_length < 2:
        raise ValueError("max_length must be >= 2")

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
            k = min(beam_width, len(valid))
            choices = valid[np.argsort(row[valid])[-k:][::-1]]
            for index in choices:
                tokens, log_score, _ = hypothesis
                candidates.append((
                    tokens + [int(index)],
                    log_score + float(np.log(row[index])),
                    int(index) == eos,
                ))
        candidates.sort(key=lambda h: h[1], reverse=True)
        hypotheses = candidates[:beam_width]

    best = max(hypotheses, key=lambda h: h[1])
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


def decode_image(assets, image_id, beam_widths=(3, 5)):
    """Caption one known Flickr8k image with both decoding methods."""
    if image_id not in assets.row_of:
        raise KeyError(f"Missing VGG16 features for {image_id}")
    if image_id not in assets.references:
        raise KeyError(f"Missing Flickr8k reference captions for {image_id}")
    vector = assets.features[assets.row_of[image_id]]

    t0 = time.perf_counter()
    greedy = greedy_decode(assets.model, vector, assets.tokenizer, assets.max_length)
    greedy_time = time.perf_counter() - t0
    beams, timings = {}, {}
    for k in beam_widths:
        t0 = time.perf_counter()
        beams[f"beam_{k}"] = beam_search_decode(
            assets.model, vector, assets.tokenizer, assets.max_length, beam_width=k
        )
        timings[f"beam_{k}"] = time.perf_counter() - t0

    return {
        "image_id": image_id,
        "references": [" ".join(_reference_tokens(c)) for c in assets.references[image_id]],
        "greedy": greedy,
        **beams,
        "seconds": {"greedy": round(greedy_time, 3),
                    **{key: round(val, 3) for key, val in timings.items()}},
    }


def example_captions(assets, num_examples=6, seed=42, beam_widths=(3, 5),
                     split="test"):
    """Reproducibly sample held-out images and generate real model predictions."""
    if split not in ("test", "val"):
        raise ValueError("Use validation or held-out test images for examples")
    if num_examples <= 0:
        raise ValueError("num_examples must be positive")
    ids = assets.splits[split]
    if not ids:
        raise ValueError(f"No images in {split} split")
    chosen = random.Random(seed).sample(ids, min(num_examples, len(ids)))
    return [decode_image(assets, image_id, beam_widths) for image_id in chosen]


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
        lines += ["", "Greedy: " + item["greedy"]]
        lines += [f"{name.replace('_', ' ').title()}: {value}"
                  for name, value in item.items() if name.startswith("beam_")]
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


def evaluate_bleu(assets, beam_widths=(3, 5), max_images=None):
    """CORPUS BLEU-1..4 on the saved held-out test split (all 5 references).

    max_images is for a quick *exploratory* run only. Final reported values
    should use all test images with max_images=None.
    """
    from nltk.translate.bleu_score import corpus_bleu

    test_ids = assets.splits["test"]
    if max_images is not None:
        if max_images <= 0:
            raise ValueError("max_images must be positive")
        test_ids = test_ids[:max_images]
    names = ["greedy", *[f"beam_{k}" for k in beam_widths]]
    predictions = {name: [] for name in names}
    references = []
    durations = {name: [] for name in names}

    for i, image_id in enumerate(test_ids, start=1):
        item = decode_image(assets, image_id, beam_widths=beam_widths)
        references.append([_reference_tokens(c) for c in assets.references[image_id]])
        for name in names:
            predictions[name].append(item[name].split())
            durations[name].append(item["seconds"][name])
        if i % 100 == 0:
            print(f"Captioned {i}/{len(test_ids)} test images", flush=True)

    weights = {
        "BLEU-1": (1., 0., 0., 0.),
        "BLEU-2": (.5, .5, 0., 0.),
        "BLEU-3": (1/3, 1/3, 1/3, 0.),
        "BLEU-4": (.25, .25, .25, .25),
    }
    metrics = {}
    for name in names:
        metrics[name] = {
            key: float(corpus_bleu(references, predictions[name], weights=weight))
            for key, weight in weights.items()
        }
        metrics[name]["avg_seconds_per_caption"] = float(np.mean(durations[name]))
    return {
        "split": "test",
        "images_evaluated": len(test_ids),
        "full_test_set": len(test_ids) == len(assets.splits["test"]),
        "scores": metrics,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Flickr8k greedy and beam-search decoding")
    parser.add_argument("--model-dir", default="models/lstm",
                        help="Folder containing trained caption_model.keras and tokenizer/splits")
    parser.add_argument("--features-dir", default=str(DEFAULT_FEATURES_DIR))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default="models/lstm/decoding_results")
    parser.add_argument("--num-examples", type=int, default=6)
    parser.add_argument("--beam-widths", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-test", action="store_true",
                        help="Also calculate corpus BLEU on held-out test images")
    parser.add_argument("--eval-limit", type=int, default=None,
                        help="Exploratory BLEU only: limit number of test images")
    args = parser.parse_args(argv)
    widths = tuple(dict.fromkeys(args.beam_widths))
    if any(k < 1 for k in widths):
        parser.error("Every beam width must be >= 1")

    assets = load_assets(args.model_dir, args.features_dir, args.data_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    examples = example_captions(assets, args.num_examples, args.seed, widths)
    for item in examples:
        print(f"\n{item['image_id']}")
        print("Reference:", item["references"][0])
        print("Greedy:   ", item["greedy"])
        for k in widths:
            print(f"Beam {k}:   ", item[f"beam_{k}"])
    (output / "caption_examples.json").write_text(
        json.dumps(examples, indent=2), encoding="utf-8"
    )
    save_example_figure(assets, examples, output / "caption_examples.png")
    print(f"\nSaved examples to {output.resolve()}")

    if args.eval_test:
        result = evaluate_bleu(assets, widths, args.eval_limit)
        (output / "bleu_comparison.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print("Corpus BLEU results:", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
