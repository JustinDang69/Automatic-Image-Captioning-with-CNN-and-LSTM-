"""Train the Flickr8k caption decoder (docs/updated_architecture.md).

Image branch: saved, frozen VGG16 fc2 (4096) -> Dense(256)
Text branch:  padded caption tokens -> Embedding(256) -> LSTM(256)
Fusion:       Add -> Dense(256) -> Softmax(vocabulary) -> next word

Run from repository root:
  python -m src.feature_extraction
  python -m src.train_lstm --epochs 20 --batch-size 128
  python -m src.train_lstm --output-dir /content/drive/MyDrive/.../lstm_run1   # Colab + Drive

Training is resumable: every epoch the full model (weights + optimizer state),
the training history and the early-stopping / learning-rate counters are saved
to ``output_dir``. Re-running the same command continues from the last
finished epoch, so a disconnected Colab session loses at most one epoch.

Greedy/beam search and BLEU evaluation are downstream, not training steps.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path

import keras
import numpy as np
from keras import Input, Model, layers

from src.data_preprocessing import CaptionTokenizer, create_sequences, prepare_flickr8k, subset_captions
from src.feature_extraction import DEFAULT_FEATURES_DIR, FEATURE_DIM, load_features

BEST_MODEL_FILE = "best_caption_model.keras"   # lowest validation loss so far
LAST_MODEL_FILE = "last_checkpoint.keras"      # most recent epoch, used to resume
FINAL_MODEL_FILE = "caption_model.keras"       # copy of the best model when training ends
STATE_FILE = "training_state.json"


# Purpose: Supplies mini-batches with aligned image features, padded caption prefixes and next-token labels.
# Implementation: Image vectors are fetched by index for each batch, avoiding a full duplicated image-feature array for every caption prefix.
class CaptionBatches(keras.utils.PyDataset):
    """Get VGG16 vectors only for caption prefixes in the current batch.

    Avoids materialising a huge (number_of_prefixes, 4096) image array.

    Each item is ``({"image_features": (B, 4096) float32,
                     "caption_input":  (B, max_length) int32}, targets (B,) int32)``.
    ``rows[k]`` is the row of ``features`` that belongs to training pair k.
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


# Purpose: Constructs a two-input caption decoder using precomputed VGG16 features and token prefixes.
# Implementation: The visual branch projects 4,096 features to 256 units; the text branch converts token IDs into embeddings and an LSTM state.
# Design rationale: Element-wise fusion requires matching vector widths. The final softmax predicts a distribution over the vocabulary.
def build_caption_model(vocab_size: int, max_length: int, learning_rate=0.001,
                        embedding_dim=256, lstm_units=256, dropout=0.4,
                        mask_padding=False):
    """Diagram-exact 256-unit branches/fusion with a vocabulary softmax.

    Input images have already passed through frozen VGG16. Only the layers
    defined here are trained.

    Padding and the GPU (``mask_padding``):
        Our captions are LEFT-padded with 0 (``[0, 0, startseq, a, dog]``, see
        ``create_sequences``). On a GPU, Keras 3 runs the LSTM with NVIDIA
        cuDNN, which only accepts masks for RIGHT-padded sequences; a left-padded
        mask makes training stop with "You are passing a RNN mask that does not
        correspond to right-padded sequences".

        * ``mask_padding=False`` (default): no mask. The LSTM reads the padding
          positions too, learns an embedding for index 0, and uses the fast
          cuDNN kernel on GPU. Because padding is on the left, the real words
          are always the last steps the LSTM sees before its output.
        * ``mask_padding=True``: padding is masked (ignored), and cuDNN is
          disabled so the mask is allowed. Same results in principle, but
          several times slower on GPU.

        Input data and the model layers are identical either way.
    """
    image_input = Input(shape=(FEATURE_DIM,), name="image_features")
    image = layers.Dropout(dropout, name="image_dropout")(image_input)
    image = layers.Dense(256, activation="relu", name="image_dense_256")(image)

    caption_input = Input(shape=(max_length,), dtype="int32", name="caption_input")
    text = layers.Embedding(vocab_size, embedding_dim, mask_zero=mask_padding,
                            name="embedding_256")(caption_input)
    text = layers.Dropout(dropout, name="caption_dropout")(text)
    text = layers.LSTM(lstm_units, use_cudnn=False if mask_padding else "auto",
                       name="lstm_256")(text)

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


# Purpose: prepare batches; see the function docstring for parameters, outputs and validation behavior.
def prepare_batches(prepared, features, row_of, split, batch_size=128, seed=42,
                    max_images=None):
    """Fit labels/tokenizer on train only; train/val use identical vocabulary.

    ``max_images`` keeps only the first N images of the split (quick smoke tests).
    Returns ``(CaptionBatches, number_of_next_word_pairs)``.
    """
    if split not in ("train", "val"):
        raise ValueError("Only train and val splits are used during fitting")
    split_ids = prepared.splits[split][:max_images]
    ids, prefixes, targets = create_sequences(
        subset_captions(prepared.captions, split_ids), prepared.tokenizer, prepared.max_length
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


# Purpose: plot training loss; see the function docstring for parameters, outputs and validation behavior.
def plot_training_loss(history, path):
    """Save a report-ready training/validation loss graph."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history["loss"]) + 1)
    ax.plot(epochs, history["loss"], label="Train loss", marker="o", markersize=3)
    ax.plot(epochs, history["val_loss"], label="Validation loss", marker="o", markersize=3)
    ax.set(xlabel="Epoch", ylabel="Cross-entropy loss",
           title="Flickr8k: LSTM training and validation loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Resumable checkpoints
# ---------------------------------------------------------------------------

# Purpose:  atomic write text; see the function docstring for parameters, outputs and validation behavior.
def _atomic_write_text(path: Path, text: str) -> None:
    """Write via a temporary file + rename, so a crash never leaves a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# Purpose:  atomic save model; see the function docstring for parameters, outputs and validation behavior.
def _atomic_save_model(model, path: Path) -> None:
    tmp = path.with_name("tmp_" + path.name)   # must still end in .keras
    model.save(tmp)
    os.replace(tmp, path)


# Purpose: load training state; see the function docstring for parameters, outputs and validation behavior.
def load_training_state(output_dir) -> dict | None:
    """Return the saved training state, or None if this is a fresh run."""
    path = Path(output_dir) / STATE_FILE
    if not path.is_file() or not (Path(output_dir) / LAST_MODEL_FILE).is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# Purpose: Maintains checkpoint, validation and callback state across interrupted training runs.
# Implementation: Persisting the model and progress after each epoch allows training to continue after a runtime disconnection.
class ResumableTraining(keras.callbacks.Callback):
    """Checkpointing, early stopping and learning-rate reduction that survive restarts.

    Keras' own EarlyStopping / ReduceLROnPlateau reset their counters every
    time ``fit`` starts, so after a Colab disconnect they would forget how many
    epochs had passed without improvement. This callback keeps those counters
    in ``training_state.json`` instead. At the end of every epoch it:

      1. records the epoch's losses/accuracies and learning rate;
      2. if ``val_loss`` improved: saves ``best_caption_model.keras``;
         otherwise counts a "bad" epoch and halves the learning rate after
         ``lr_patience`` bad epochs (never below ``min_lr``);
      3. stops training after ``patience`` bad epochs in a row;
      4. saves ``last_checkpoint.keras`` (weights + optimizer) and then the
         state, history (JSON + CSV) and loss graph.

    Only validation loss is used for these decisions; the test split is never seen.
    """

    def __init__(self, output_dir, config, patience=4, lr_patience=2, lr_factor=0.5,
                 min_lr=1e-6, state=None):
        super().__init__()
        self.out = Path(output_dir)
        self.config = config
        self.patience, self.lr_patience = patience, lr_patience
        self.lr_factor, self.min_lr = lr_factor, min_lr
        state = state or {}
        self.history = state.get("history", {})
        self.best_val_loss = state.get("best_val_loss", math.inf)
        self.best_epoch = state.get("best_epoch", 0)
        self.wait = state.get("wait", 0)          # epochs since last improvement
        self.lr_wait = state.get("lr_wait", 0)    # bad epochs since last LR change
        self.stopped_early = state.get("stopped_early", False)

    def on_epoch_end(self, epoch, logs=None):
        logs = dict(logs or {})
        logs["learning_rate"] = float(self.model.optimizer.learning_rate)
        for key, value in logs.items():
            self.history.setdefault(key, []).append(float(value))

        val_loss = logs["val_loss"]
        if not math.isfinite(val_loss):
            raise RuntimeError(f"Epoch {epoch + 1}: val_loss is {val_loss}; try a lower learning rate.")
        if val_loss < self.best_val_loss:
            self.best_val_loss, self.best_epoch = val_loss, epoch + 1
            self.wait = self.lr_wait = 0
            _atomic_save_model(self.model, self.out / BEST_MODEL_FILE)
            print(f"\nEpoch {epoch + 1}: val_loss improved to {val_loss:.4f}; saved best model.")
        else:
            self.wait += 1
            self.lr_wait += 1
            if self.lr_wait >= self.lr_patience:
                old_lr = logs["learning_rate"]
                new_lr = max(old_lr * self.lr_factor, self.min_lr)
                if new_lr < old_lr:
                    self.model.optimizer.learning_rate = new_lr
                    print(f"\nEpoch {epoch + 1}: reducing learning rate {old_lr:.2e} -> {new_lr:.2e}.")
                self.lr_wait = 0
            if self.wait >= self.patience:
                self.stopped_early = True
                self.model.stop_training = True
                print(f"\nEpoch {epoch + 1}: early stopping (best epoch {self.best_epoch}).")

        # Checkpoint first, then the state that points to it.
        _atomic_save_model(self.model, self.out / LAST_MODEL_FILE)
        state = {
            "epochs_completed": epoch + 1,
            "best_val_loss": self.best_val_loss, "best_epoch": self.best_epoch,
            "wait": self.wait, "lr_wait": self.lr_wait, "stopped_early": self.stopped_early,
            "config": self.config, "history": self.history,
        }
        _atomic_write_text(self.out / STATE_FILE, json.dumps(state, indent=1))
        save_history(self.history, self.out)


# Purpose: save history; see the function docstring for parameters, outputs and validation behavior.
def save_history(history, out: Path) -> None:
    """Write history.json, training_history.csv and training_loss.png."""
    _atomic_write_text(out / "history.json", json.dumps(history, indent=2))
    keys = list(history)
    with open(out / "training_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", *keys])
        for i in range(len(history["loss"])):
            writer.writerow([i + 1, *(history[k][i] for k in keys)])
    plot_training_loss(history, out / "training_loss.png")


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

# Purpose: Loads prepared captions and cached visual features, configures batches, then fits the caption decoder.
# Implementation: The saved checkpoint is selected using validation performance rather than the final training epoch.
def train(data_dir=None, features_dir=DEFAULT_FEATURES_DIR, output_dir="models/lstm",
          epochs=20, batch_size=128, learning_rate=0.001, dropout=0.4,
          patience=4, seed=42, min_count=1, resume=True, max_images=None,
          mask_padding=False, verbose=1):
    """Train (or resume training) the caption decoder.

    Returns ``(best_model, history, prepared)`` where ``history`` holds every
    epoch run so far (including epochs from earlier, interrupted sessions).
    ``max_images`` limits the number of train AND val images (smoke tests only).
    """
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size, and patience must be positive")
    if learning_rate <= 0 or not 0 <= dropout < 1 or min_count < 1:
        raise ValueError("Invalid learning rate, dropout or min_count")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = load_training_state(out) if resume else None
    start_epoch = state["epochs_completed"] if state else 0
    keras.utils.set_random_seed(seed + start_epoch)

    # 1. Flickr8k captions, split by image (80/10/10) and training-only tokenizer.
    prepared = prepare_flickr8k(data_dir=data_dir, seed=seed, min_count=min_count)
    print("Image split sizes:", {k: len(v) for k, v in prepared.splits.items()})
    print(f"Vocabulary: {prepared.vocab_size} | Max length: {prepared.max_length}")

    config = {
        "architecture": "VGG16(fc2, frozen) + Dense256 | Embedding256 + LSTM256 | Add + Dense256 + softmax",
        "epochs_requested": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
        "dropout": dropout, "patience": patience, "seed": seed, "min_count": min_count,
        "mask_padding": mask_padding, "max_images": max_images,
        "vocab_size": prepared.vocab_size, "max_length": prepared.max_length,
    }
    if state:
        _check_resume_compatible(state["config"], config, out, prepared.tokenizer)
        print(f"Resuming from {out / LAST_MODEL_FILE} after epoch {start_epoch}.")
    prepared.save(out)   # tokenizer.json, splits.json, metadata.json beside the model

    # 2. Pre-extracted frozen VGG16 fc2 vectors; VGG16 is NOT retrained.
    #    Loaded fully into RAM (~133 MB): random per-batch reads from a memory-mapped
    #    file on Google Drive would be very slow.
    features, row_of = load_features(features_dir, require_complete=True, mmap=False)
    if features.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected feature dim {FEATURE_DIM}, got {features.shape[1]}")
    train_batches, n_train = prepare_batches(
        prepared, features, row_of, "train", batch_size, seed + start_epoch, max_images
    )
    val_batches, n_val = prepare_batches(
        prepared, features, row_of, "val", batch_size, seed, max_images
    )
    print(f"Next-word examples: {n_train:,} train / {n_val:,} validation")
    config.update(train_pairs=n_train, val_pairs=n_val)

    # 3. Two-input caption decoder trained by teacher forcing.
    if state:
        model = keras.models.load_model(out / LAST_MODEL_FILE)
    else:
        model = build_caption_model(prepared.vocab_size, prepared.max_length,
                                    learning_rate=learning_rate, dropout=dropout,
                                    mask_padding=mask_padding)
        if verbose:
            model.summary()

    progress = ResumableTraining(out, config, patience=patience,
                                 lr_patience=max(2, patience // 2), state=state)
    if progress.stopped_early or start_epoch >= epochs:
        print(f"Nothing to train: {start_epoch} epochs done"
              f"{' (stopped early)' if progress.stopped_early else ''}.")
    else:
        model.fit(train_batches, validation_data=val_batches, initial_epoch=start_epoch,
                  epochs=epochs, callbacks=[progress], verbose=verbose)

    # 4. Persist the best decoder and the exact vocabulary/splits needed later.
    shutil.copyfile(out / BEST_MODEL_FILE, out / FINAL_MODEL_FILE)
    history = progress.history
    summary = {**config, "epochs_run": len(history["loss"]),
               "best_epoch": progress.best_epoch, "best_val_loss": progress.best_val_loss,
               "stopped_early": progress.stopped_early}
    _atomic_write_text(out / "training_config.json", json.dumps(summary, indent=2))
    print(f"Saved models, tokenizer, splits and training plots to {out.resolve()}")
    print(f"Best epoch {progress.best_epoch}: val_loss {progress.best_val_loss:.4f}")
    print("Held-out test images have NOT been used for training or model selection.")
    return keras.models.load_model(out / BEST_MODEL_FILE), history, prepared


# Purpose:  check resume compatible; see the function docstring for parameters, outputs and validation behavior.
def _check_resume_compatible(saved, current, out, tokenizer):
    """Refuse to resume if the data/model settings changed since the checkpoint."""
    # batch_size and epochs may change between sessions; these may not.
    keys = ("vocab_size", "max_length", "seed", "min_count", "mask_padding", "max_images",
            "dropout")
    changed = {k: (saved.get(k), current[k]) for k in keys if saved.get(k) != current[k]}
    saved_tok = out / "tokenizer.json"
    if saved_tok.is_file() and CaptionTokenizer.load(saved_tok).word_index != tokenizer.word_index:
        changed["tokenizer"] = "vocabulary differs"
    if changed:
        raise ValueError(
            f"Cannot resume in {out}: settings changed since the checkpoint {changed}. "
            "Use a new output_dir or pass resume=False (--no-resume) to start again.")


# Purpose: main; see the function docstring for parameters, outputs and validation behavior.
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
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Ignore checkpoints in output-dir and start from epoch 0")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Use only the first N train and val images (smoke test)")
    parser.add_argument("--mask-padding", action="store_true",
                        help="Mask padding in the LSTM (disables cuDNN; slower on GPU)")
    args = parser.parse_args(argv)
    train(**vars(args))


if __name__ == "__main__":
    main()
