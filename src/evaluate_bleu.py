import numpy as np
import json
import sys
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu
from decoding_enhancements import length_normalize_score, top_k_sampling

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

def load_artifacts():
    print("Loading model and features...")
    model = load_model(os.path.join(PROJECT_ROOT, 'models', 'best_caption_model.keras'))
    
    tokenizer_path = os.path.join(PROJECT_ROOT, 'models', 'tokenizer.json')
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer()
    
    if 'word_index' in data:
        raw_word_index = data['word_index']
    elif 'config' in data and 'word_index' in data['config']:
        raw_word_index = data['config']['word_index']
    else:
        raw_word_index = data
        
    if isinstance(raw_word_index, str):
        raw_word_index = json.loads(raw_word_index)
        
    tokenizer.word_index = raw_word_index
    tokenizer.index_word = {int(v): str(k) for k, v in raw_word_index.items()}
        
    features = np.load(os.path.join(PROJECT_ROOT, 'data', 'features', 'features.npy'))
    with open(os.path.join(PROJECT_ROOT, 'data', 'features', 'image_ids.json'), 'r', encoding='utf-8') as f:
        image_ids = json.load(f)
        
    feature_dict = {img_id: features[i] for i, img_id in enumerate(image_ids)}
    return model, tokenizer, feature_dict

def generate_caption(model, tokenizer, photo_feature, max_length, strategy='greedy', k=3):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        yhat = model.predict([np.array([photo_feature]), sequence], verbose=0)[0]
        
        if strategy == 'top_k':
            word_id = top_k_sampling(yhat, k=k)
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
            
            top_k_indices = np.argsort(yhat)[-beam_width:]
            
            for word_id in top_k_indices:
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

def evaluate_model(model, tokenizer, features, test_dict, max_length, strategy='greedy'):
    actual, predicted = list(), list()
    
    print(f"\nEvaluating {strategy.upper()} strategy...")
    for img_id, ref_captions in test_dict.items():
        if img_id not in features:
            continue
            
        if strategy in ['greedy', 'top_k']:
            yhat = generate_caption(model, tokenizer, features[img_id], max_length, strategy=strategy)
        else:
            yhat = beam_search_caption(model, tokenizer, features[img_id], max_length, strategy=strategy)
            
        references = [ref.split() for ref in ref_captions]
        actual.append(references)
        predicted.append(yhat.split())
        
    print(f"BLEU-1: {corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)):.4f}")
    print(f"BLEU-2: {corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)):.4f}")
    print(f"BLEU-3: {corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0)):.4f}")
    print(f"BLEU-4: {corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)):.4f}")

if __name__ == '__main__':
    model, tokenizer, feature_dict = load_artifacts()
    max_len = 39 
    
    print("Loading test splits and references...")
    with open(os.path.join(PROJECT_ROOT, 'data', 'processed', 'splits.json'), 'r', encoding='utf-8') as f:
        splits = json.load(f)
    
    test_image_ids = set(splits['test'])
    
    test_data = {}
    with open(os.path.join(PROJECT_ROOT, 'data', 'flickr8k', 'captions.txt'), 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
        for line in lines[1:]: 
            if len(line) < 2:
                continue
            image_id, caption = line.split(',', 1)
            
            if image_id in test_image_ids:
                if image_id not in test_data:
                    test_data[image_id] = []
                clean_caption = caption.lower().strip()
                test_data[image_id].append(clean_caption)
    
 
    # evaluate_modsave time, comment out the two we already calculatedel(model, tokenizer, feature_dict, test_data, max_len, strategy='greedy')
    # evaluate_model(model, tokenizer, feature_dict, test_data, max_len, strategy='top_k')
    
    # Run the remaining evaluations
    evaluate_model(model, tokenizer, feature_dict, test_data, max_len, strategy='beam_search')
    evaluate_model(model, tokenizer, feature_dict, test_data, max_len, strategy='length_norm')