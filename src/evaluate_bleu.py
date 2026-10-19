import argparse
import json
import os
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from src.caption_decoding import load_assets, _reference_tokens, greedy_decode, beam_search_decode

def evaluate_model(assets, test_dict, strategy='top_k', beam_width=3, top_k=3, alpha=0.7, seed=42):
    actual, predicted = [], []
    rng = np.random.default_rng(seed)
    
    print(f"\nEvaluating {strategy.upper()} strategy...")
    for img_id, ref_captions in test_dict.items():
        vector = assets.features[assets.row_of[img_id]]
        
        if strategy == 'top_k':
            yhat = greedy_decode(assets.model, vector, assets.tokenizer, assets.max_length, top_k=top_k, rng=rng)
        elif strategy == 'length_norm':
            yhat = beam_search_decode(assets.model, vector, assets.tokenizer, assets.max_length, beam_width=beam_width, alpha=alpha)
            
        actual.append([ref.split() for ref in ref_captions])
        predicted.append(yhat.split())
        
    weights = {
        "BLEU-1": (1.0, 0, 0, 0),
        "BLEU-2": (0.5, 0.5, 0, 0),
        "BLEU-3": (1/3, 1/3, 1/3, 0),
        "BLEU-4": (0.25, 0.25, 0.25, 0.25)
    }
    
    scores = {}
    for name, w in weights.items():
        scores[name] = round(float(corpus_bleu(actual, predicted, weights=w)), 4)
        print(f"{name}: {scores[name]:.4f}")
        
    return scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate BLEU for Top-k and Length-Norm")
    parser.add_argument("--model-dir", default="models/lstm", help="Path to trained model and tokenizer")
    parser.add_argument("--features-dir", default="data/features", help="Path to VGG16 features")
    parser.add_argument("--output-json", default="bleu_results.json", help="Path to save results")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading test splits and references using baseline loader...")
    assets = load_assets(args.model_dir, args.features_dir)
    
    test_data = {}
    for img_id in assets.splits['test']:
        if img_id in assets.references:
            test_data[img_id] = [" ".join(_reference_tokens(ref)) for ref in assets.references[img_id]]
            
    results = {
        "model_path": args.model_dir,
        "image_count": len(test_data),
        "top_k": args.top_k,
        "beam_width": args.beam_width,
        "alpha": args.alpha,
        "seed": args.seed,
        "scores": {}
    }
    
    results["scores"]["top_k"] = evaluate_model(assets, test_data, 'top_k', top_k=args.top_k, seed=args.seed)
    results["scores"]["length_norm"] = evaluate_model(assets, test_data, 'length_norm', beam_width=args.beam_width, alpha=args.alpha)
        
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nReproducible evaluation saved to {args.output_json}")