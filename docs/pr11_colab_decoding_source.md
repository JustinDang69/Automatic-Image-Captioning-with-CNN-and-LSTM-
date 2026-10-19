# Google Colab: PR #11 decoding-enhancement source and testing

Open [notebooks/06_colab_decoding_enhancements.ipynb](../notebooks/06_colab_decoding_enhancements.ipynb) on `feature/pr11-colab-decoding-source`.

This branch starts at **PR #11** (`fix/decoding-enhancements-review`) and adds a runnable notebook. The notebook includes a **verbatim copy** of PR #11's `src/decoding_enhancements.py` in an editable code cell, then explicitly binds the existing `src.caption_decoding.greedy_decode` and `beam_search_decode` to those notebook-local functions. The model/padding and decoder logic stay imported from PR #11.

## Steps

1. In Colab, open the notebook from this branch. Change `RUN_NAME` or `MODEL_DIR_OVERRIDE` for the folder containing `caption_model.keras` or `best_caption_model.keras`, `tokenizer.json`, `splits.json`, and `metadata.json`.
2. Check `FEATURES_DIR_OVERRIDE` if your VGG16 `features.npy`, `image_ids.json`, `progress.npy`, and `metadata.json` live outside `MyDrive/NIT3251_captioning/features`.
3. Run cells in order. The setup clones **this PR #11-derived branch**, runs decoding tests, then executes the exact copied enhancement source. After editing the source, rerun its cell and the immediately following binding cell.
4. With the trained model available, Section 9 prints actual **greedy, beam, top-k sampling and length-normalised beam** captions for selected Flickr8k validation images. Section 10 displays the images and reference captions. Optionally run validation BLEU, then the final held-out test BLEU once your parameters are fixed.
5. Captions and figures are saved to the model run's `decoding_enhancements` folder.

PR #11 and `main` are unchanged; review the new branch/PR before merging. No private `.keras` weights are included. No trained-model caption output or BLEU values were generated in this session.
