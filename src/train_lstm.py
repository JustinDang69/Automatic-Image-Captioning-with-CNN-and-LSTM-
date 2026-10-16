"""Train the Flickr8k caption decoder shown in mermaid-diagram.png.

Image branch: saved, frozen VGG16 fc2 (4096) -> Dense(256)
Text branch:  padded caption tokens -> Embedding(256) -> LSTM(256)
Fusion:       Add -> Dense(256) -> Softmax(vocabulary) -> next word

Run from repository root:
  python -m src.feature_extraction
  python -m src.train_lstm --epochs 20 --batch-size 128

Greedy/beam search and BLEU evaluation are downstream, not training steps.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import keras
import numpy as np
from keras import Input, Model, layers
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src.data_preprocessing import create_sequences, prepare_flickr8k
from src.feature_extraction import DEFAULT_FEATURES_DIR, FEATURE_DIM, load_features


class CaptionBatches(keras.utils.PyDataset):
    """Get VGG16 vectors only for caption prefixes in the current batch.

    Avoids materialising a huge (number_of_prefixes, 4096) image array.
    """

    def __init__(self, features, rows, prefixes, targets, batch_size=128,
                 shuffle=True, seed=42):
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not (len(rows) == len(prefixes) == len(targets)):
            raise ValueError("Rows, prefixes and next-word targets must align")
        self.features = features
        self.rows = np.asarray(rows, dtype=np.int32)
        self.prefixes = np.asarray(prefixes, dtype=np.int32)
        self.targets = np.asarray(targets, dtype=np.int32)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(self.targets))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.order) / self.batch_size)

    def __getitem__(self, index):
        indices = self.order[index * self.batch_size:(index + 1) * self.batch_size]
        image_batch = np.asarray(self.features[self.rows[indices]], dtype=np.float32)
        return {
            "image_features": image_batch,
            "caption_input": self.prefixes[indices],
        }, self.targets[indices]

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)


def build_caption_model(vocab_size: int, max_length: int, learning_rate=0.001,
                        embedding_dim=256, lstm_units=256, dropout=0.4):
    """Diagram-exact 256-unit branches/fusion with a vocabulary softmax.

    Input images have already passed through frozen VGG16. Only the layers
    defined here are trained. Padding index zero is masked for the LSTM.
    The current repository uses left-padding; it works but may be slower
    on GPU than right-padded masked sequences.
    """
    image_input = Input(shape=(FEATURE_DIM,), name="image_features")
    image = layers.Dropout(dropout, name="image_dropout")(image_input)
    image = layers.Dense(256, activation="relu", name="image_dense_256")(image)

    caption_input = Input(shape=(max_length,), dtype="int32", name="caption_input")
    text = layers.Embedding(vocab_size, embedding_dim, mask_zero=True,
                            name="embedding_256")(caption_input)
    text = layers.Dropout(dropout, name="caption_dropout")(text)
    text = layers.LSTM(lstm_units, name="lstm_256")(text)

    merged = layers.Add(name="feature_fusion")([image, text])
    merged = layers.Dense(256, activation="relu", name="fusion_dense_256")(merged)
    merged = layers.Dropout(dropout, name="fusion_dropout")(merged)
    next_word = layers.Dense(vocab_size, activation="softmax",
                             name="next_word_softmax")(merged)

    model = Model([image_input, caption_input], next_word, name="vgg16_lstm_captioner")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model


def prepare_batches(prepared, features, row_of, split, batch_size=128, seed=42):
    """Fit labels/tokenizer on train only; train/val use identical vocabulary."""
    if split not in ("train", "val"):
        raise ValueError("Only train and val splits are used during fitting")
    ids, prefixes, targets = create_sequences(
        prepared.split_captions(split), prepared.tokenizer, prepared.max_length
    )
    if not len(targets):
        raise ValueError(f"No next-word examples in {split} split")
    missing = sorted(set(ids) - row_of.keys())
    if missing:
        raise ValueError(f"No VGG16 features for {len(missing)} {split} images: {missing[:3]}")
    rows = np.fromiter((row_of[i] for i in ids), dtype=np.int32, count=len(ids))
    if targets.min() < 1 or targets.max() >= prepared.vocab_size:
        raise ValueError("Target word indices are outside the tokenizer vocabulary")
    return CaptionBatches(features, rows, prefixes, targets, batch_size,
                          shuffle=(split == "train"), seed=seed), len(targets)


def plot_training_loss(history, path):
    """Save a report-ready training/validation loss graph."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["loss"], label="Train loss")
    ax.plot(history["val_loss"], label="Validation loss")
    ax.set(xlabel="Epoch", ylabel="Cross-entropy loss",
           title="Flickr8k: LSTM training and validation loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def train(data_dir=None, features_dir=DEFAULT_FEATURES_DIR, output_dir="models/lstm",
          epochs=20, batch_size=128, learning_rate=0.001, dropout=0.4,
          patience=4, seed=42, min_count=1):
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size, and patience must be positive")
    if learning_rate <= 0 or not 0 <= dropout < 1 or min_count < 1:
        raise ValueError("Invalid learning rate, dropout or min_count")

    keras.utils.set_random_seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Flickr8k captions, split by image (80/10/10) and training-only tokenizer.
    prepared = prepare_flickr8k(data_dir=data_dir, seed=seed, min_count=min_count)
    prepared.save(out)
    print("Image split sizes:", {k: len(v) for k, v in prepared.splits.items()})
    print(f"Vocabulary: {prepared.vocab_size} | Max length: {prepared.max_length}")

    # 2. Pre-extracted frozen VGG16 fc2 vectors; VGG16 is NOT retrained.
    features, row_of = load_features(features_dir, require_complete=True, mmap=True)
    if features.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected feature dim {FEATURE_DIM}, got {features.shape[1]}")
    train_batches, n_train = prepare_batches(
        prepared, features, row_of, "train", batch_size, seed
    )
    val_batches, n_val = prepare_batches(
        prepared, features, row_of, "val", batch_size, seed
    )
    print(f"Next-word examples: {n_train:,} train / {n_val:,} validation")

    # 3. Two-input caption decoder trained by teacher forcing.
    model = build_caption_model(prepared.vocab_size, prepared.max_length,
                                learning_rate=learning_rate, dropout=dropout)
    model.summary()
    callbacks = [
        ModelCheckpoint(str(out / "best_caption_model.keras"), monitor="val_loss",
                        mode="min", save_best_only=True),
        EarlyStopping(monitor="val_loss", mode="min", patience=patience,
                      restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=max(2, patience // 2), min_lr=1e-6),
        CSVLogger(str(out / "training_history.csv")),
    ]
    result = model.fit(train_batches, validation_data=val_batches,
                       epochs=epochs, callbacks=callbacks)

    # 4. Persist decoder and the exact vocabulary/splits needed later.
    model.save(out / "caption_model.keras")
    history = {k: [float(v) for v in values] for k, values in result.history.items()}
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_training_loss(history, out / "training_loss.png")
    metadata = {
        "architecture": "VGG16(fc2, frozen) + Dense256 | Embedding256 + LSTM256 | Add + Dense256 + softmax",
        "epochs_requested": epochs, "epochs_run": len(history["loss"]),
        "batch_size": batch_size, "learning_rate": learning_rate,
        "dropout": dropout, "patience": patience, "seed": seed,
        "vocab_size": prepared.vocab_size, "max_length": prepared.max_length,
        "train_pairs": n_train, "val_pairs": n_val,
        "best_epoch": int(np.argmin(history["val_loss"]) + 1),
        "best_val_loss": float(min(history["val_loss"])),
    }
    (out / "training_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved models, tokenizer, splits and training plots to {out.resolve()}")
    print("Held-out test images have NOT been used for training or model selection.")
    print("Next: greedy/beam decoding and BLEU evaluation on the test split.")
    return model, result, prepared


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the Flickr8k LSTM caption decoder")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--features-dir", default=str(DEFAULT_FEATURES_DIR))
    parser.add_argument("--output-dir", default="models/lstm")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-count", type=int, default=1)
    args = parser.parse_args(argv)
    train(**vars(args))


if __name__ == "__main__":
    main()
