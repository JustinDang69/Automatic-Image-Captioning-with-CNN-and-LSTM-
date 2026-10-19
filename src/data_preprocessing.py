"""Flickr8k data preparation for the CNN (VGG16) + LSTM image-captioning model.

Pipeline overview
-----------------
1. ``get_dataset_paths``      locate ``Images/`` and ``captions.txt``.
2. ``load_captions``          parse ``captions.txt`` into {image_id: [caption, ...]}.
3. ``verify_image_files``     check every annotated image exists on disk.
4. ``clean_captions``         lowercase, strip punctuation, add startseq/endseq.
5. ``split_image_ids``        80/10/10 split by *image*, never by caption.
6. ``CaptionTokenizer.fit``   build the vocabulary from TRAINING captions only.
7. ``get_max_length``         longest training caption (in tokens).
8. ``create_sequences``       turn each caption into (prefix -> next word) pairs.

``prepare_flickr8k`` runs steps 1-7 in one call.

Expected dataset layout (Kaggle "adityajn105/flickr8k")::

    data/flickr8k/
    ├── Images/          8,091 .jpg files
    └── captions.txt     CSV with header "image,caption", 5 captions per image

The dataset folder can be changed with the ``FLICKR8K_DIR`` environment
variable or the ``data_dir`` argument, so the same code runs on Windows,
WSL2 and Google Colab without editing any paths.

Image pixels and VGG16 features are NOT loaded here. Training pairs only store
the image ID, so the 4,096-d feature vector for an image can be looked up once
per batch instead of being copied for every word in every caption.
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from keras.utils import pad_sequences

START_TOKEN = "startseq"
END_TOKEN = "endseq"
PAD_INDEX = 0          # index 0 is reserved for padding by pad_sequences / Embedding(mask_zero=True)
OOV_TOKEN = "<unk>"    # index 1: any word not seen in the training captions

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "flickr8k"

DOWNLOAD_HELP = (
    "Download Flickr8k from https://www.kaggle.com/datasets/adityajn105/flickr8k "
    "and extract it so that 'Images/' and 'captions.txt' sit inside data/flickr8k/ "
    "(or set the FLICKR8K_DIR environment variable to the extracted folder). "
    "See README.md for details."
)


# ---------------------------------------------------------------------------
# 1-3. Loading the raw dataset
# ---------------------------------------------------------------------------

def get_dataset_paths(data_dir: str | os.PathLike | None = None) -> tuple[Path, Path]:
    """Return ``(images_dir, captions_file)`` and check that both exist.

    The folder is chosen in this order: the ``data_dir`` argument, the
    ``FLICKR8K_DIR`` environment variable, then ``<project>/data/flickr8k``.

    Raises:
        FileNotFoundError: if the folder, ``Images/`` or ``captions.txt`` is missing.
    """
    root = Path(data_dir or os.environ.get("FLICKR8K_DIR") or DEFAULT_DATA_DIR)
    images_dir = root / "Images"
    captions_file = root / "captions.txt"

    if not root.is_dir():
        raise FileNotFoundError(f"Flickr8k folder not found: {root}\n{DOWNLOAD_HELP}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image folder not found: {images_dir}\n{DOWNLOAD_HELP}")
    if not captions_file.is_file():
        raise FileNotFoundError(f"Caption file not found: {captions_file}\n{DOWNLOAD_HELP}")
    return images_dir, captions_file


# Q&A: Read Flickr8k caption annotations and group the five reference captions by image ID. WHY: Image IDs keep descriptions tied to the correct photo.
def load_captions(captions_file: str | os.PathLike) -> dict[str, list[str]]:
    """Parse ``captions.txt`` into a mapping ``{image_id: [raw caption, ...]}``.

    The file is a CSV with the header ``image,caption``. Captions that contain
    commas are wrapped in double quotes, e.g.::

        1012212859_01547e3f17.jpg,"A dog shakes its head near the shore , a red ball next to it ."

    so the ``csv`` module is used instead of a naive ``line.split(",")``,
    which would cut such captions in half.

    Each image appears on several rows (5 in Flickr8k); all of its captions
    are collected into one list, keeping the file order.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the header is wrong or a row is malformed/empty.
    """
    captions_file = Path(captions_file)
    if not captions_file.is_file():
        raise FileNotFoundError(f"Caption file not found: {captions_file}\n{DOWNLOAD_HELP}")

    captions: dict[str, list[str]] = {}
    with open(captions_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None or [h.strip().lower() for h in header] != ["image", "caption"]:
            raise ValueError(
                f"{captions_file}: expected header 'image,caption' but found {header!r}. "
                "Is this the Kaggle Flickr8k captions.txt?"
            )
        for line_no, row in enumerate(reader, start=2):
            if not row:
                continue  # skip blank lines
            if len(row) != 2 or not row[0].strip() or not row[1].strip():
                raise ValueError(f"{captions_file}, line {line_no}: malformed row {row!r}")
            image_id, caption = row[0].strip(), row[1].strip()
            captions.setdefault(image_id, []).append(caption)

    if not captions:
        raise ValueError(f"{captions_file} contains no captions.")
    return captions


def verify_image_files(image_ids, images_dir: str | os.PathLike) -> None:
    """Check that every image ID in the caption file has a matching file.

    Raises:
        FileNotFoundError: listing (the first few) missing images.
    """
    images_dir = Path(images_dir)
    missing = sorted(i for i in image_ids if not (images_dir / i).is_file())
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} image(s) listed in captions.txt are missing from "
            f"{images_dir}: {preview}{' ...' if len(missing) > 5 else ''}"
        )


# ---------------------------------------------------------------------------
# 4. Caption cleaning
# ---------------------------------------------------------------------------

# Q&A: Normalize words before training. WHY: Fewer spelling/case variants mean a more consistent vocabulary.
def clean_caption(caption: str) -> str:
    """Normalise one caption and wrap it in ``startseq ... endseq``.

    Steps:
      1. lowercase;
      2. delete apostrophes so "isn't" -> "isnt" (Flickr8k also writes
         possessives as a separate "'s" token, e.g. "dog 's");
      3. replace every other character that is not a-z / 0-9 with a space
         ("tri-colored" -> "tri colored", "ball ." -> "ball");
      4. drop leftover single letters such as the "s" from "dog 's",
         but KEEP "a" (a meaningful word) and single digits;
      5. collapse whitespace and add the start/end tokens.

    Example:
        >>> clean_caption("A dog 's red ball , tri-colored !")
        'startseq a dog red ball tri colored endseq'

    ``startseq`` tells the decoder where to begin and ``endseq`` teaches it
    when to stop generating.
    """
    text = caption.lower().replace("'", "")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    words = [w for w in text.split() if len(w) > 1 or w == "a" or w.isdigit()]
    return " ".join([START_TOKEN, *words, END_TOKEN])


def clean_captions(captions: dict[str, list[str]]) -> dict[str, list[str]]:
    """Apply ``clean_caption`` to every caption, keeping the image -> captions grouping."""
    return {image_id: [clean_caption(c) for c in caps] for image_id, caps in captions.items()}


# ---------------------------------------------------------------------------
# 5. Train / validation / test split (by image)
# ---------------------------------------------------------------------------

# Q&A: Split unique IMAGE IDs (not caption rows) into train/validation/test. WHY: All five captions of an image stay in one split, preventing data leakage.
def split_image_ids(
    image_ids,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Split unique image IDs into train / val / test lists.

    Why split by image and not by caption?
        Each image has 5 captions. If captions were split independently, the
        same photo could appear in training (with caption 1) and in testing
        (with caption 2). The model would then be "tested" on an image it has
        already seen, inflating BLEU scores - this is data leakage. Splitting
        unique image IDs guarantees all 5 captions of a photo stay together in
        exactly one split.

    Reproducibility:
        IDs are sorted first, then shuffled with ``random.Random(seed)``, so the
        result depends only on the set of IDs and the seed - not on file order
        or on any global random state.

    Sizes use floor for train and val; the test set takes the remainder
    (Flickr8k: 8,091 -> 6,472 / 809 / 810).
    """
    if not (0 < train_frac < 1 and 0 <= val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("train_frac and val_frac must be in (0, 1) and sum to less than 1.")

    ids = sorted(set(image_ids))
    random.Random(seed).shuffle(ids)

    n_train = int(len(ids) * train_frac)
    n_val = int(len(ids) * val_frac)
    return {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }


def subset_captions(captions: dict[str, list[str]], image_ids) -> dict[str, list[str]]:
    """Return the captions for the given image IDs only (e.g. one split)."""
    return {image_id: captions[image_id] for image_id in image_ids}


# ---------------------------------------------------------------------------
# 6-7. Tokenizer and maximum caption length
# ---------------------------------------------------------------------------

# Q&A: Build a word-to-integer dictionary from TRAINING captions only. WHY: Neural networks need numeric tokens; fitting on training data avoids test leakage.
class CaptionTokenizer:
    """Word <-> integer mapping built from training captions.

    Index layout:
        0 -> padding (never a real word)
        1 -> ``<unk>`` (any word not in the training vocabulary)
        2.. -> training words, most frequent first

    ``vocab_size`` (= number of words + 2) is the size the model's Embedding
    input and softmax output layer must use.

    A small custom class is used because ``keras.preprocessing.text.Tokenizer``
    was removed in Keras 3. This class keeps the same idea (``word_index``,
    ``texts_to_sequences``) and saves to plain JSON so the exact vocabulary can
    be reloaded later for training, greedy/beam decoding and evaluation.
    """

    def __init__(self, min_count: int = 1):
        self.min_count = min_count
        self.word_index: dict[str, int] = {}
        self.index_word: dict[int, str] = {}

    def fit(self, texts) -> "CaptionTokenizer":
        """Build the vocabulary. Pass TRAINING captions only.

        Words appearing fewer than ``min_count`` times are left out and will be
        encoded as ``<unk>``. Ties in frequency are broken alphabetically so the
        vocabulary is identical on every run.
        """
        counts = Counter(word for text in texts for word in text.split())
        words = sorted(
            (w for w, c in counts.items() if c >= self.min_count),
            key=lambda w: (-counts[w], w),
        )
        self.word_index = {OOV_TOKEN: 1}
        for i, word in enumerate(words, start=2):
            self.word_index[word] = i
        self.index_word = {i: w for w, i in self.word_index.items()}
        return self

    @property
    def vocab_size(self) -> int:
        """Number of output classes for the model (+1 for the padding index 0)."""
        return len(self.word_index) + 1

    def texts_to_sequences(self, texts) -> list[list[int]]:
        """Convert cleaned captions to lists of word IDs; unseen words -> ``<unk>`` (1)."""
        oov = self.word_index[OOV_TOKEN]
        return [[self.word_index.get(w, oov) for w in text.split()] for text in texts]

    def sequences_to_texts(self, sequences) -> list[str]:
        """Convert word-ID lists back to text, skipping padding (0)."""
        return [
            " ".join(self.index_word[int(i)] for i in seq if int(i) != PAD_INDEX)
            for seq in sequences
        ]

    def save(self, path: str | os.PathLike) -> None:
        """Save the vocabulary as JSON."""
        data = {"min_count": self.min_count, "word_index": self.word_index}
        Path(path).write_text(json.dumps(data, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | os.PathLike) -> "CaptionTokenizer":
        """Load a vocabulary saved with ``save``."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(min_count=data["min_count"])
        tok.word_index = data["word_index"]
        tok.index_word = {i: w for w, i in tok.word_index.items()}
        return tok


def get_max_length(captions: dict[str, list[str]]) -> int:
    """Length (in words, including startseq/endseq) of the longest caption.

    Compute this on the TRAINING split; it fixes the padded input length of
    the LSTM decoder and the maximum number of decoding steps at inference.
    """
    return max(len(c.split()) for caps in captions.values() for c in caps)


# ---------------------------------------------------------------------------
# 8. Next-word training sequences
# ---------------------------------------------------------------------------

# Q&A: Turn each N-token caption into N-1 next-word examples. seq[:i] = words already seen; seq[i] = target next word; pad on the LEFT to a fixed length.
def create_sequences(
    captions: dict[str, list[str]],
    tokenizer: CaptionTokenizer,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn every caption into (input prefix -> next word) training pairs.

    For ``startseq a dog runs endseq`` (IDs ``seq``), the loop below produces::

        i=1  seq[:1] = startseq              -> seq[1] = a
        i=2  seq[:2] = startseq a            -> seq[2] = dog
        i=3  seq[:3] = startseq a dog        -> seq[3] = runs
        i=4  seq[:4] = startseq a dog runs   -> seq[4] = endseq

    i.e. ``seq[:i]`` is the input prefix (everything before position i) and
    ``seq[i]`` is the next-word target. Prefixes are left ("pre") padded with
    0 to ``max_length`` so the real words sit next to the LSTM's final step.
    Captions longer than ``max_length`` (possible in val/test) keep their most
    recent words ("pre" truncation).

    Returns:
        image_ids: (N,)             str   - which image each pair belongs to
        X:         (N, max_length)  int32 - padded word-ID prefixes
        y:         (N,)             int32 - next-word IDs in [1, vocab_size)

    ``y`` holds integer IDs rather than one-hot vectors: with ~380k pairs and
    a vocabulary of ~8k words, one-hot targets would need ~12 GB. Integer
    targets work directly with ``sparse_categorical_crossentropy``.

    Image features are not copied here. During training, look up
    ``features[image_ids[k]]`` (a 4,096-d VGG16 vector) per batch.
    """
    pair_ids, prefixes, targets = [], [], []
    for image_id, caps in captions.items():
        for seq in tokenizer.texts_to_sequences(caps):
            for i in range(1, len(seq)):
                pair_ids.append(image_id)
                prefixes.append(seq[:i])   # input: words before position i
                targets.append(seq[i])     # target: the word at position i

    X = pad_sequences(prefixes, maxlen=max_length, padding="pre", truncating="pre", dtype="int32")
    return np.array(pair_ids), X, np.array(targets, dtype=np.int32)


# ---------------------------------------------------------------------------
# One-call pipeline
# ---------------------------------------------------------------------------

@dataclass
class PreparedData:
    """Everything later stages need from data preparation."""

    captions: dict[str, list[str]]   # cleaned captions for ALL images
    splits: dict[str, list[str]]     # {"train"|"val"|"test": [image_id, ...]}
    tokenizer: CaptionTokenizer      # fitted on training captions only
    vocab_size: int
    max_length: int
    images_dir: Path

    def split_captions(self, split: str) -> dict[str, list[str]]:
        """Cleaned captions for one split ("train", "val" or "test")."""
        return subset_captions(self.captions, self.splits[split])

    def save(self, output_dir: str | os.PathLike) -> None:
        """Write splits, tokenizer and metadata to ``output_dir`` for later tasks."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "splits.json").write_text(json.dumps(self.splits, indent=1), encoding="utf-8")
        self.tokenizer.save(out / "tokenizer.json")
        meta = {"vocab_size": self.vocab_size, "max_length": self.max_length,
                "start_token": START_TOKEN, "end_token": END_TOKEN, "oov_token": OOV_TOKEN}
        (out / "metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


# Q&A: Orchestrate caption loading, cleaning, splitting, tokenization and prefix/target preparation. Explain the sequence of these steps to the tutor.
def prepare_flickr8k(
    data_dir: str | os.PathLike | None = None,
    seed: int = 42,
    min_count: int = 1,
) -> PreparedData:
    """Run loading, validation, cleaning, splitting and tokenizer fitting.

    The tokenizer and ``max_length`` are computed from the training split only,
    so no information from validation/test captions leaks into the model.
    """
    images_dir, captions_file = get_dataset_paths(data_dir)
    raw = load_captions(captions_file)
    verify_image_files(raw.keys(), images_dir)

    captions = clean_captions(raw)
    splits = split_image_ids(captions.keys(), seed=seed)
    train_caps = subset_captions(captions, splits["train"])

    tokenizer = CaptionTokenizer(min_count=min_count).fit(
        c for caps in train_caps.values() for c in caps
    )
    return PreparedData(
        captions=captions,
        splits=splits,
        tokenizer=tokenizer,
        vocab_size=tokenizer.vocab_size,
        max_length=get_max_length(train_caps),
        images_dir=images_dir,
    )
