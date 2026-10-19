# Colab decoding-enhancements notebook

Use [notebooks/06_colab_decoding_enhancements.ipynb](../notebooks/06_colab_decoding_enhancements.ipynb) on branch `feature/colab-decoding-enhancements` to test the **PR #7** implementation, without modifying your trained CNN–LSTM architecture.

## How to run

1. In Google Colab select **File → Open notebook → GitHub** and choose this branch and `06_colab_decoding_enhancements.ipynb` (or download the notebook from the GitHub file page and upload it).
2. Set `RUN_NAME` or `MODEL_DIR_OVERRIDE` to the **folder** containing `caption_model.keras` (or `best_caption_model.keras`), `tokenizer.json`, `metadata.json`, `splits.json`. These files are **not in Git**.
3. Set `FEATURES_DIR_OVERRIDE` to the folder containing `features.npy`, `image_ids.json`, `progress.npy`, `metadata.json`. Alternatively keep the existing `MyDrive/NIT3251_captioning/features` location.
4. Use `DATA_DIR_OVERRIDE` if Flickr8k is available on Drive; otherwise KaggleHub will download images and captions.
5. Run the cells in order. Dependencies and the two PR #7 unit tests run before model inference.
6. Use `EXAMPLE_SPLIT="val"` for tuning. Adjust `TOP_K`, `BEAM_WIDTH`, and `ALPHA`, then optionally run validation BLEU.
7. Once settings are fixed, set `RUN_FINAL_TEST_BLEU=True` for a corpus BLEU comparison across the full held-out test split.

The notebook shows **greedy**, **standard beam**, **top-k sampling**, and **length-normalised beam** captions from the **same** trained model, displaying real Flickr8k images and all five references. It writes PNG/JSON caption examples and, when evaluation flags are enabled, CSV/JSON corpus BLEU results and a chart to the run's `decoding_enhancements` folder.

PR #7's old `bleu_evaluation_charts.ipynb` attempts to import `load_artifacts`, `generate_caption`, and `beam_search_caption`, which do not exist in `src.evaluate_bleu`. The new notebook avoids those incompatible imports and uses `src.caption_decoding.load_assets`, `greedy_decode`, `beam_search_decode` directly.

**Limitations:** No end-to-end Colab evaluation can be demonstrated without your private trained model and VGG16 feature files. The source notebook has been checked for JSON structure but not executed against the actual Flickr8k checkpoint. Top-k sampling is stochastic even with fixed settings; use consistent seeds to compare runs.
