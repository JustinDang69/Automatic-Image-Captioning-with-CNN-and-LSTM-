"""Corpus BLEU-1..4 for the two decoding innovations on the held-out test split.

Uses exactly the same assets and protocol as the verified greedy / standard-beam
baseline (``src.caption_decoding``): the trained ``caption_model.keras`` from the
run folder, its ``tokenizer.json`` / ``metadata.json`` (max_length) / ``splits.json``,
the saved VGG16 features, all five references per image cleaned with
``clean_caption``, and BLEU-3 weights of exactly (1/3, 1/3, 1/3, 0).

Run from the repository root (as a module, so ``src`` can be imported):
    python -m src.evaluate_bleu --model-dir <run dir> --features-dir <features dir> \\
        --data-dir <Flickr8k dir> --top-k 3 --seed 42 --beam-width 3 --alpha 0.7 \\
        --output-json <run dir>/decoding_results/innovations.json
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from src.caption_decoding import load_assets, _reference_tokens, greedy_decode, beam_search_decode
from src.decoding_enhancements import validate_top_k

METHODS = ("top_k", "length_norm")


# Purpose: Evaluates generated captions across a held-out test split against multiple human reference captions.
# Implementation: Corpus BLEU-1 through BLEU-4 use increasing n-gram orders; these scores measure reference overlap rather than factual correctness.
def evaluate_model(assets, test_dict, strategy='top_k', beam_width=3, top_k=3, alpha=0.7, seed=42,
                   captions=None):
    """Decode every image in ``test_dict`` with one innovation and return BLEU-1..4.

    ``test_dict`` maps image_id -> five cleaned reference strings. For Top-k, ONE
    generator ``np.random.default_rng(seed)`` is shared by all images (in test-split
    order), so the run is reproducible and each word gets a new random draw.
    If ``captions`` is a dict, the generated caption for each image is stored in it.
    """
    if strategy not in METHODS:
        raise ValueError(f"Unknown strategy {strategy!r}; expected one of {METHODS}")
    actual, predicted = [], []
    rng = np.random.default_rng(seed)

    print(f"\nEvaluating {strategy.upper()} strategy...")
    for i, (img_id, ref_captions) in enumerate(test_dict.items(), 1):
        vector = assets.features[assets.row_of[img_id]]

        if strategy == 'top_k':
            yhat = greedy_decode(assets.model, vector, assets.tokenizer, assets.max_length, top_k=top_k, rng=rng)
        elif strategy == 'length_norm':
            yhat = beam_search_decode(assets.model, vector, assets.tokenizer, assets.max_length, beam_width=beam_width, alpha=alpha)

        actual.append([ref.split() for ref in ref_captions])
        predicted.append(yhat.split())
        if captions is not None:
            captions[img_id] = yhat
        if i % 100 == 0:
            print(f"Decoded {i}/{len(test_dict)} test images", flush=True)

    weights = {
        "BLEU-1": (1.0, 0, 0, 0),
        "BLEU-2": (0.5, 0.5, 0, 0),
        "BLEU-3": (1/3, 1/3, 1/3, 0),
        "BLEU-4": (0.25, 0.25, 0.25, 0.25)
    }

    scores = {}
    for name, w in weights.items():
        # Full precision, like the baseline's bleu_comparison.json; printed to 4 d.p.
        scores[name] = float(corpus_bleu(actual, predicted, weights=w))
        print(f"{name}: {scores[name]:.4f}")

    return scores


# Purpose:  loaded model file; see the function docstring for parameters, outputs and validation behavior.
def _loaded_model_file(model_dir):
    """The checkpoint load_assets() actually loads (caption_model.keras preferred)."""
    model_dir = Path(model_dir)
    preferred = model_dir / "caption_model.keras"
    return str(preferred if preferred.exists() else model_dir / "best_caption_model.keras")


# Purpose:  training summary; see the function docstring for parameters, outputs and validation behavior.
def _training_summary(model_dir):
    """best_epoch / best_val_loss from training_config.json, if the run folder has it."""
    path = Path(model_dir) / "training_config.json"
    if not path.is_file():
        return {}
    config = json.loads(path.read_text(encoding="utf-8"))
    return {k: config[k] for k in ("best_epoch", "best_val_loss") if k in config}


# Purpose: main; see the function docstring for parameters, outputs and validation behavior.
def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate BLEU for Top-k and Length-Norm")
    parser.add_argument("--model-dir", default="models/lstm", help="Path to trained model and tokenizer")
    parser.add_argument("--features-dir", default="data/features", help="Path to VGG16 features")
    parser.add_argument("--data-dir", default=None,
                        help="Flickr8k folder with Images/ and captions.txt "
                             "(default: FLICKR8K_DIR environment variable, then data/flickr8k)")
    parser.add_argument("--output-json", default="bleu_results.json", help="Path to save results")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS),
                        help="Which innovations to evaluate (default: both)")
    args = parser.parse_args(argv)
    try:
        validate_top_k(args.top_k)
    except ValueError as e:
        parser.error(str(e))
    if args.beam_width < 1:
        parser.error("--beam-width must be >= 1")
    if args.alpha < 0:
        parser.error("--alpha must be >= 0")

    print("Loading test splits and references using baseline loader...")
    assets = load_assets(args.model_dir, args.features_dir, args.data_dir)

    test_data = {}
    for img_id in assets.splits['test']:
        if img_id in assets.references:
            test_data[img_id] = [" ".join(_reference_tokens(ref)) for ref in assets.references[img_id]]

    results = {
        "model_path": args.model_dir,
        "model_file": _loaded_model_file(args.model_dir),
        **_training_summary(args.model_dir),
        "features_dir": str(args.features_dir),
        "data_dir": str(args.data_dir or os.environ.get("FLICKR8K_DIR") or "data/flickr8k"),
        "split": "test",
        "image_count": len(test_data),
        "full_test_set": len(test_data) == len(assets.splits["test"]),
        "max_length": assets.max_length,
        "vocab_size": assets.tokenizer.vocab_size,
        "top_k": args.top_k,
        "beam_width": args.beam_width,
        "alpha": args.alpha,
        "seed": args.seed,
        "methods": {
            "top_k": {"decoder": "greedy_decode(top_k, rng=np.random.default_rng(seed))",
                      "top_k": args.top_k, "seed": args.seed},
            "length_norm": {"decoder": "beam_search_decode(beam_width, alpha)",
                            "beam_width": args.beam_width, "alpha": args.alpha},
        },
        "scores": {},
        "empty_captions": {},
        "captions": {},
    }

    for method in args.methods:
        captions = {}
        if method == "top_k":
            results["scores"]["top_k"] = evaluate_model(assets, test_data, 'top_k', top_k=args.top_k, seed=args.seed, captions=captions)
        else:
            results["scores"]["length_norm"] = evaluate_model(assets, test_data, 'length_norm', beam_width=args.beam_width, alpha=args.alpha, captions=captions)
        results["captions"][method] = captions
        results["empty_captions"][method] = sum(not c.strip() for c in captions.values())

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    print(f"\nReproducible evaluation saved to {args.output_json}")
    return results


if __name__ == '__main__':
    main()
