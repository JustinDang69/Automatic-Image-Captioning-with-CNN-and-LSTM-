# Automatic-Image-Captioning-with-CNN-and-LSTM-
Link to the dataset: https://www.kaggle.com/datasets/adityajn105/flickr8k/data

NIT3251 Assessment 2: a CNN (VGG16) encoder + LSTM decoder that generates captions for Flickr8k images.

## Project structure

```text
├── README.md
├── requirements.txt
├── pytest.ini
├── mermaid-diagram.png              # architecture diagram
├── src/
│   └── data_preprocessing.py        # Flickr8k loading, cleaning, splitting, tokenizer, sequences
├── notebooks/
│   └── 01_data_preparation.ipynb    # runs and explains the data pipeline
├── tests/
│   └── test_data_preprocessing.py
└── data/                            # NOT in Git (see below)
    ├── flickr8k/{Images/, captions.txt}
    └── processed/                   # splits.json, tokenizer.json, metadata.json
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
```

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
