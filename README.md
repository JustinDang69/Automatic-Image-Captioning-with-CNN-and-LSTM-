# Automatic-Image-Captioning-with-CNN-and-LSTM-
Link to the dataset: https://www.kaggle.com/datasets/adityajn105/flickr8k/data

NIT3251 Assessment 2: a CNN (VGG16) encoder + LSTM decoder that generates captions for Flickr8k images.

## Project structure

```text
├── README.md
├── requirements.txt
├── pytest.ini
├── docs/updated_architecture.md    # current CNN–LSTM and decoding diagram
├── src/
│   ├── data_preprocessing.py        # Flickr8k loading, cleaning, splitting, tokenizer, sequences
│   ├── feature_extraction.py        # frozen VGG16 (fc2) -> 4,096-d feature per image
│   └── train_lstm.py                # train separate LSTM decoder from saved features
├── notebooks/
│   ├── 01_data_preparation.ipynb    # runs and explains the data pipeline
│   ├── 02_vgg16_feature_extraction.ipynb
│   ├── 03_lstm_training.ipynb       # train, validate and plot decoder learning
│   └── 04_colab_lstm_training.ipynb # Colab GPU training with Google Drive checkpoints
├── tests/
│   ├── test_data_preprocessing.py
│   ├── test_feature_extraction.py
│   └── test_train_lstm.py
└── data/                            # NOT in Git (see below)
    ├── flickr8k/{Images/, captions.txt}
    ├── processed/                   # splits.json, tokenizer.json, metadata.json
    └── features/                    # features.npy (8091, 4096), image_ids.json, metadata.json
```

## Setup (Windows / VS Code)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

TensorFlow 2.11+ has no native-Windows GPU support, so it runs on CPU locally.
Data preparation runs fine on CPU; for GPU training use Google Colab or WSL2.

## Getting the dataset

The dataset (~1 GB) is not stored in this repository. Either:

- **Manual:** download the zip from the Kaggle link above and extract it so you have
  `data/flickr8k/Images/` (8,091 `.jpg` files) and `data/flickr8k/captions.txt`.
- **Script:**
  ```powershell
  python -c "import kagglehub; kagglehub.dataset_download('adityajn105/flickr8k', output_dir='data/flickr8k')"
  ```

If the dataset lives somewhere else (e.g. Colab), set `FLICKR8K_DIR` to the folder
that contains `Images/` and `captions.txt`.

## Running

```powershell
python -m pytest -v                                     # unit tests (full-dataset tests skip if data is missing)
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_preparation.ipynb
python -m src.feature_extraction                        # VGG16 features, ~20 min on CPU, resumable
```

The first VGG16 run downloads the ImageNet weights (~528 MB) to `~/.keras/models/`.

## Data pipeline summary

| Step | Function | Result on Flickr8k |
|---|---|---|
| Load captions | `load_captions` | 8,091 images, 40,455 captions (5 per image) |
| Clean | `clean_caption` | lowercase, no punctuation, `startseq ... endseq` |
| Split by image (seed 42) | `split_image_ids` | 6,472 train / 809 val / 810 test images |
| Tokenizer (train only) | `CaptionTokenizer` | vocab size 7,702 (incl. padding 0 and `<unk>`) |
| Max length (train) | `get_max_length` | 39 tokens |
| Next-word pairs | `create_sequences` | train X `(381317, 39)`, y `(381317,)` |

The Kaggle version has no official split files, so the 80/10/10 split uses a fixed seed.

## VGG16 features

`python -m src.feature_extraction` passes each image once through a frozen ImageNet VGG16
(input 224 x 224 x 3, `vgg16.preprocess_input`) and saves the 4,096-d `fc2` output.
If interrupted, re-run the same command and it continues where it stopped.

```python
from src.feature_extraction import load_features
features, row_of = load_features()                       # (8091, 4096) float32, {image_id: row}
vec = features[row_of["1000268201_693b08cb0e.jpg"]]      # (4096,)
```


## Train the LSTM caption decoder (diagram: Dense256 + Embedding256/LSTM256)

After `python -m src.feature_extraction` completes and creates `data/features/features.npy`, train the **decoder only**:

```powershell
python -m src.train_lstm --epochs 20 --batch-size 128
```

Or open `notebooks/03_lstm_training.ipynb` in VS Code or Jupyter and execute its cells in order. For GPU training on Google Colab, open `notebooks/04_colab_lstm_training.ipynb` (see below). VGG16 stays frozen; its saved 4,096-dimensional feature vectors feed the Dense(256) image branch. Captions are tokenised and padded into an Embedding(256) → LSTM(256) branch. The two 256-dimensional outputs are added, followed by Dense(256) and a vocabulary softmax next-word predictor. Loss is sparse categorical cross-entropy; the optimiser is Adam. Training uses only the training images and model selection uses validation images.

Outputs under `models/lstm/` (or `--output-dir`): `best_caption_model.keras`, `caption_model.keras` (copy of the best model), `last_checkpoint.keras`, `training_state.json`, `tokenizer.json`, `splits.json`, `metadata.json`, `training_history.csv`, `history.json`, `training_config.json` and `training_loss.png`.

### Resuming and Google Colab

Training saves `last_checkpoint.keras` (weights + optimizer) and `training_state.json` (epoch, early-stopping
and learning-rate counters, history) after every epoch. Re-running the same command resumes from the last
finished epoch; `--no-resume` starts again. Resuming is refused if the vocabulary or key settings changed.

On Colab, `notebooks/04_colab_lstm_training.ipynb` mounts Google Drive and saves every epoch to
`MyDrive/NIT3251_captioning/runs/<RUN_NAME>/`. Upload `data/features/{features.npy, image_ids.json,
progress.npy, metadata.json}` to `MyDrive/NIT3251_captioning/features/` first; VGG16 is not re-run.

The LSTM uses cuDNN on GPU. cuDNN cannot use a padding mask with our left-padded captions, so padding is not
masked by default (`--mask-padding` enables a mask but disables cuDNN, which is much slower).

**Next:** Implement greedy and beam-search caption generation and compute corpus BLEU on the reserved test images. Training accuracy alone is *not* the caption BLEU score.
