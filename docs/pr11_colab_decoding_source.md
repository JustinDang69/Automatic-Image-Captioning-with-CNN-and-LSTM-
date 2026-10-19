# Colab notebook: two decoding enhancements

Open [notebooks/06_colab_decoding_enhancements.ipynb](../notebooks/06_colab_decoding_enhancements.ipynb) on the `main` branch (the notebook's `REPO_BRANCH` is `main`). It was added by [PR #12](https://github.com/JustinDang69/Automatic-Image-Captioning-with-CNN-and-LSTM-/pull/12), built on PR #11's decoding source; both are merged into `main`.

The notebook includes an **exact, editable copy** of PR #11's `src/decoding_enhancements.py`. It runs **only** the following two strategies on the saved CNN–LSTM and VGG16 features:

- **Top-k sampling:** `top_k=3`, `RANDOM_SEED=42`
- **Length-normalised beam:** `beam_width=3`, `alpha=0.7`

The notebook does **not** produce separate greedy or unnormalised beam baselines. The original PR #11 greedy/beam functions remain the internal inference implementations.

## Google Colab steps

1. Open the notebook from `main`. Point `MODEL_DIR_OVERRIDE` (or `RUN_NAME`) at the folder containing `caption_model.keras` or `best_caption_model.keras`, `tokenizer.json`, `splits.json`, `metadata.json`.
2. If necessary, set `FEATURES_DIR_OVERRIDE` to the folder with `features.npy`, `image_ids.json`, `progress.npy`, and `metadata.json`. Provide Flickr8k through `DATA_DIR_OVERRIDE`, or let KaggleHub download it.
3. Execute cells in order. Source-code and binding cells use the notebook's copied PR #11 functions. Tests run before inference.
4. **Section 9** prints two actual model captions for each chosen image; **Section 10** displays images and all five reference captions. Example PNG/JSON are saved under the model run's `decoding_enhancements` folder.
5. **Section 12** offers an optional 20-image validation smoke test; it is disabled by default.
6. **Sections 13–14 run by default** on all **810 held-out test images** with `RUN_FINAL_TEST_BLEU=True`, producing a full corpus BLEU-1–4 table, two-series chart, and **per-image captions for all 810 images** as CSV and JSON. Results are saved as CSV/JSON and an updated plot. This may require substantial runtime.

The existing trained model is required; the repository does not contain model weights or feature arrays.

## Completed 810-image evaluation (19 October 2026)

The full evaluation has been run in Google Colab on the trained `lstm_run1` checkpoint (best epoch 5,
validation loss 3.3437) and the saved VGG16 features, over **all 810 held-out test images**, each scored
against its five cleaned references with NLTK corpus BLEU. The model was not retrained.

| Method | Parameters | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---|---|---:|---:|---:|---:|
| Top-k sampling | k = 3, seed 42 | 0.4553 | 0.2571 | 0.1400 | 0.0771 |
| Length-normalised beam | width 3, alpha 0.7 | 0.5395 | 0.3508 | 0.2236 | 0.1469 |

For comparison, the baselines on the same 810 images were greedy 0.5420 / 0.3488 / 0.2141 / 0.1350 and
standard beam (width 3) 0.5402 / 0.3480 / 0.2200 / 0.1428; see the README's *Final test-set results*.
The result files (BLEU CSV/JSON and per-image captions) are in the private Drive run folder and are not
committed to Git.
