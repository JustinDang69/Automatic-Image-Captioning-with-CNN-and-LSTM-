import numpy as np
import json
import sys
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu
from decoding_enhancements import length_normalize_score, top_k_sampling

from caption_decoding import load_assets, _reference_tokens

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))


def generate_caption(model, tokenizer, photo_feature, max_length, strategy='greedy', top_k=3):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        yhat = model.predict([np.array([photo_feature]), sequence], verbose=0)[0]
        
        if strategy == 'top_k':
            word_id = top_k_sampling(yhat, top_k= top_k)
        else: 
            word_id = int(np.argmax(yhat))
            
        word = tokenizer.index_word.get(word_id)
        if word is None:
            break
            
        in_text += ' ' + word
        if word == 'endseq':
            break
            
    return in_text.replace('startseq ', '').replace(' endseq', '').strip()

def beam_search_caption(model, tokenizer, photo_feature, max_length, beam_width=3, strategy='beam_search', alpha=0.7):
    start_id = tokenizer.word_index.get('startseq')
    end_id = tokenizer.word_index.get('endseq')
    
    # Track beams as (sequence_of_ids, cumulative_log_prob)
    beams = [([start_id], 0.0)]
    
    for _ in range(max_length):
        new_beams = []
        for seq, score in beams:
            if seq[-1] == end_id:
                new_beams.append((seq, score))
                continue
                
            padded_seq = pad_sequences([seq], maxlen=max_length)
            yhat = model.predict([np.array([photo_feature]), padded_seq], verbose=0)[0]
            
            beam_indices = np.argsort(yhat)[-beam_width:]
            
            for word_id in beam_indices:
                prob = yhat[word_id]
                log_prob = np.log(prob + 1e-10) # Avoid log(0)
                new_score = score + log_prob
                new_beams.append((seq + [word_id], new_score))
                
        # Sort and prune based on strategy
        if strategy == 'length_norm':
            new_beams.sort(key=lambda x: length_normalize_score(x[1], len(x[0]), alpha), reverse=True)
        else:
            new_beams.sort(key=lambda x: x[1], reverse=True)
            
        beams = new_beams[:beam_width]
        if all(beam[0][-1] == end_id for beam in beams):
            break
            
    best_seq = beams[0][0]
    words = [tokenizer.index_word.get(idx) for idx in best_seq if idx not in [start_id, end_id] and idx is not None]
    return ' '.join(words)

def evaluate_model(model, tokenizer, features, test_dict, max_length, strategy='greedy', beam_width=3, top_k=3, alpha=0.7):
    actual, predicted = list(), list()
    
    print(f"\nEvaluating {strategy.upper()} strategy...")
    for img_id, ref_captions in test_dict.items():
        if img_id not in features:
            continue
            
        if strategy in ['greedy', 'top_k']:
            yhat = generate_caption(model, tokenizer, features[img_id], max_length, strategy=strategy, top_k=top_k)
        else:
            yhat = beam_search_caption(model, tokenizer, features[img_id], max_length, beam_width=beam_width, strategy=strategy, alpha=alpha)
            
        references = [ref.split() for ref in ref_captions]
        actual.append(references)
        predicted.append(yhat.split())
        
    scores = {
        "BLEU-1": round(float(corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))), 4),
        "BLEU-2": round(float(corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))), 4),
        "BLEU-3": round(float(corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0))), 4),
        "BLEU-4": round(float(corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))), 4),
    }
    
    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")
        
    return scores

if __name__ == '__main__':
    print("Loading test splits and references using baseline loader...")
    # This single line replaces all your manual loading code
    assets = load_assets(os.path.join(PROJECT_ROOT, 'models'))
    
    max_len = assets.max_length
    model = assets.model
    tokenizer = assets.tokenizer
    feature_dict = {img_id: assets.features[assets.row_of[img_id]] 
                    for img_id in assets.splits['test']}
    
    test_data = {}
    for img_id in assets.splits['test']:
        if img_id in assets.references:
            # Kevan's _reference_tokens matches the project's exact text cleaning
            test_data[img_id] = [" ".join(_reference_tokens(ref)) for ref in assets.references[img_id]]
            
    results = {}
    strategies = ['greedy', 'top_k', 'beam_search', 'length_norm']
    
    for strat in strategies:
        results[strat] = evaluate_model(
            model, tokenizer, feature_dict, test_data, max_len,
            strategy=strat, beam_width=3, top_k=3, alpha=0.7
        )
        
    json_path = os.path.join(PROJECT_ROOT, 'bleu_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nReproducible evaluation saved to {json_path}")