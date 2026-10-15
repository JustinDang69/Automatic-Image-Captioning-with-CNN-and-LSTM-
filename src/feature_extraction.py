"""VGG16 image feature extraction for the CNN + LSTM captioning model.

Every Flickr8k image is passed ONCE through a frozen, ImageNet-pre-trained
VGG16 and the 4,096-d output of its ``fc2`` layer is saved. The future LSTM
training pipeline then loads these vectors instead of running VGG16 again for
each of the ~380k (prefix -> next word) training pairs.

Saved files (``data/features/``, Git-ignored)::

    features.npy     float32 array, shape (num_images, 4096); row i <-> image_ids[i]
    image_ids.json   list of image file names in row order (sorted)
    progress.npy     bool array, progress[i] is True once row i has been written
    metadata.json    model, layer, preprocessing, shapes, failed images, status

Run from the project root::

    python -m src.feature_extraction                # all images, resumable
    python -m src.feature_extraction --limit 10     # quick smoke test

Features are extracted for train, val AND test images. This does not leak
test information: VGG16 is frozen and its weights come from ImageNet, so
nothing is fitted to Flickr8k here. Only the LSTM training (next task) must be
restricted to the training split.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from src.data_preprocessing import PROJECT_ROOT, get_dataset_paths, load_captions

IMAGE_SIZE = (224, 224)          # VGG16 input resolution (height, width)
FEATURE_LAYER = "fc2"            # second 4,096-unit fully connected layer
FEATURE_DIM = 4096
DEFAULT_FEATURES_DIR = PROJECT_ROOT / "data" / "features"
PREPROCESSING = (
    "keras.applications.vgg16.preprocess_input on RGB uint8-range (0-255) float32 arrays: "
    "RGB->BGR channel swap, then subtract ImageNet BGR means (103.939, 116.779, 123.68), no scaling"
)

FEATURES_FILE = "features.npy"
IDS_FILE = "image_ids.json"
PROGRESS_FILE = "progress.npy"
METADATA_FILE = "metadata.json"


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def build_encoder(weights: str | None = "imagenet"):
    """Return a frozen VGG16 that maps (batch, 224, 224, 3) images to (batch, 4096).

    Why ``fc2``?
        VGG16 ends with fc1 (4096) -> fc2 (4096) -> predictions (1000-way
        softmax over ImageNet classes). The ``predictions`` layer only answers
        "which of 1,000 ImageNet classes is this?", which is too narrow for
        describing a scene. ``fc2`` is the last layer before that decision: a
        compact 4,096-number summary of objects, textures and layout that the
        caption decoder can learn to use.

    Why frozen?
        The ImageNet weights already encode general visual features learnt from
        1.2M images. Flickr8k (6,472 training images) is far too small to
        re-train 134M VGG16 parameters without overfitting, and a frozen encoder
        lets us compute features once and reuse them for every epoch.

    ``weights=None`` builds an untrained VGG16 (used only by tests).
    """
    from keras.applications import VGG16
    from keras.models import Model

    vgg = VGG16(weights=weights, include_top=True, input_shape=(*IMAGE_SIZE, 3))
    encoder = Model(inputs=vgg.input, outputs=vgg.get_layer(FEATURE_LAYER).output,
                    name="vgg16_fc2_encoder")
    encoder.trainable = False   # freezes every layer; no weights change in training
    return encoder


# ---------------------------------------------------------------------------
# Image loading and VGG16 preprocessing
# ---------------------------------------------------------------------------

def load_image(path: str | os.PathLike) -> np.ndarray:
    """Load one image as a float32 RGB array of shape (224, 224, 3), values 0-255.

    ``color_mode="rgb"`` converts greyscale / RGBA / palette images to 3-channel
    RGB. The image is resized directly to 224 x 224 (bilinear), which can
    slightly change the aspect ratio - the standard approach for VGG16.
    Pixel values are NOT rescaled to [0, 1]; VGG16 expects 0-255 input.
    """
    from keras.utils import img_to_array, load_img

    img = load_img(path, color_mode="rgb", target_size=IMAGE_SIZE, interpolation="bilinear")
    return img_to_array(img, dtype="float32")


def preprocess_batch(images: np.ndarray) -> np.ndarray:
    """Apply VGG16's own preprocessing to a batch of RGB 0-255 images.

    The ImageNet VGG16 weights were trained on images in "caffe" format, so
    ``vgg16.preprocess_input``:
      1. reorders channels RGB -> BGR, and
      2. subtracts the ImageNet mean pixel (B=103.939, G=116.779, R=123.68).
    It does not divide by 255. Feeding [0, 1]-scaled or BGR-loaded images would
    give the network inputs it never saw during training and weaker features.
    """
    from keras.applications.vgg16 import preprocess_input

    # copy: preprocess_input may modify float arrays in place
    return preprocess_input(np.array(images, dtype="float32", copy=True))


# ---------------------------------------------------------------------------
# Resumable extraction
# ---------------------------------------------------------------------------

def _save_progress(done: np.ndarray, path: Path) -> None:
    """Write the progress mask atomically (temp file + rename).

    An interruption mid-write leaves the previous mask intact, so at worst the
    last batch is recomputed - rows are never marked done before they are written.
    """
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.save(f, done)
    os.replace(tmp, path)


def _write_metadata(out: Path, n_images: int, done: np.ndarray, failed: dict) -> dict:
    import keras
    import tensorflow as tf

    meta = {
        "model": "VGG16",
        "weights": "imagenet",
        "layer": FEATURE_LAYER,
        "feature_dim": FEATURE_DIM,
        "input_shape": [*IMAGE_SIZE, 3],
        "resize": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} bilinear, RGB",
        "preprocessing": PREPROCESSING,
        "dtype": "float32",
        "num_images": n_images,
        "num_extracted": int(done.sum()),
        "complete": bool(done.all()),
        "failed_images": failed,
        "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__,
        "files": {"features": FEATURES_FILE, "image_ids": IDS_FILE, "progress": PROGRESS_FILE},
    }
    (out / METADATA_FILE).write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def extract_features(
    image_ids,
    images_dir: str | os.PathLike,
    output_dir: str | os.PathLike = DEFAULT_FEATURES_DIR,
    batch_size: int = 32,
    encoder=None,
    verbose: bool = True,
) -> dict:
    """Extract fc2 features for ``image_ids`` into ``output_dir``; resumable.

    How it works:
      * IDs are de-duplicated and sorted, so each image gets exactly one row and
        the row order is reproducible. ``image_ids.json`` records that order.
      * ``features.npy`` is pre-allocated on disk (memory-mapped) with shape
        (num_images, 4096). Each batch writes straight into its own rows.
      * After a batch is flushed to disk, ``progress.npy`` marks those rows done.
        Re-running the function skips rows already marked done, so an
        interruption only loses the batch in progress.
      * Images that fail to load are recorded in metadata and skipped; a later
        run retries them.

    Batching: images are stacked into (batch_size, 224, 224, 3) arrays so VGG16
    processes many images per forward pass, which is much faster than one at a time.

    Returns the metadata dict (also written to ``metadata.json``).
    """
    ids = sorted(set(image_ids))
    images_dir = Path(images_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    features_path, ids_path, progress_path = out / FEATURES_FILE, out / IDS_FILE, out / PROGRESS_FILE

    if features_path.exists():
        # Resume: only valid if the existing file was made for the same images.
        saved_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if saved_ids != ids:
            raise ValueError(
                f"{out} already holds features for a different image list "
                f"({len(saved_ids)} vs {len(ids)} images). Use another output_dir "
                "or delete the old files to start again."
            )
        features = np.lib.format.open_memmap(features_path, mode="r+")
        done = np.load(progress_path) if progress_path.exists() else np.zeros(len(ids), bool)
    else:
        ids_path.write_text(json.dumps(ids), encoding="utf-8")
        features = np.lib.format.open_memmap(
            features_path, mode="w+", dtype=np.float32, shape=(len(ids), FEATURE_DIM))
        done = np.zeros(len(ids), dtype=bool)
        _save_progress(done, progress_path)

    todo = np.flatnonzero(~done)
    if verbose:
        print(f"{len(ids)} images, {len(ids) - len(todo)} already extracted, {len(todo)} to do.")
    if len(todo) and encoder is None:
        encoder = build_encoder()

    failed: dict[str, str] = {}
    start = time.perf_counter()
    for b in range(0, len(todo), batch_size):
        rows, batch = [], []
        for row in todo[b:b + batch_size]:
            try:
                batch.append(load_image(images_dir / ids[row]))
                rows.append(row)
            except Exception as e:  # corrupt / unreadable file: record and continue
                failed[ids[row]] = f"{type(e).__name__}: {e}"
        if not rows:
            continue

        feats = np.asarray(encoder.predict_on_batch(preprocess_batch(np.stack(batch))))
        if feats.shape != (len(rows), FEATURE_DIM) or not np.isfinite(feats).all():
            raise RuntimeError(f"Bad encoder output {feats.shape} for rows {rows[:3]}...")

        features[rows] = feats
        features.flush()                 # 1) features on disk ...
        done[rows] = True
        _save_progress(done, progress_path)  # 2) ... then mark them done

        if verbose:
            n = b + len(todo[b:b + batch_size])
            rate = n / (time.perf_counter() - start)
            print(f"  {n}/{len(todo)}  {rate:.1f} img/s  ETA {(len(todo) - n) / rate / 60:.1f} min",
                  flush=True)

    del features  # release the memory map (required on Windows before reuse)
    meta = _write_metadata(out, len(ids), done, failed)
    meta["elapsed_seconds"] = round(time.perf_counter() - start, 1)
    if verbose:
        print(f"Done: {meta['num_extracted']}/{len(ids)} extracted, {len(failed)} failed, "
              f"{meta['elapsed_seconds']} s this run.")
    return meta


# ---------------------------------------------------------------------------
# Loading features in later tasks
# ---------------------------------------------------------------------------

def load_features(
    features_dir: str | os.PathLike = DEFAULT_FEATURES_DIR,
    require_complete: bool = True,
    mmap: bool = False,
) -> tuple[np.ndarray, dict[str, int]]:
    """Load saved features and an ``{image_id: row}`` lookup.

    Example (next task)::

        features, row_of = load_features()
        vec = features[row_of["1000268201_693b08cb0e.jpg"]]   # shape (4096,)

    Raises:
        FileNotFoundError: if no features have been extracted.
        ValueError: if extraction is incomplete and ``require_complete`` is True.
    """
    d = Path(features_dir)
    if not (d / FEATURES_FILE).is_file():
        raise FileNotFoundError(
            f"No features found in {d}. Run: python -m src.feature_extraction")
    ids = json.loads((d / IDS_FILE).read_text(encoding="utf-8"))
    done = np.load(d / PROGRESS_FILE)
    if require_complete and not done.all():
        raise ValueError(
            f"Feature extraction incomplete: {int(done.sum())}/{len(ids)} images. "
            "Re-run python -m src.feature_extraction to resume.")
    features = np.load(d / FEATURES_FILE, mmap_mode="r" if mmap else None)
    if features.shape != (len(ids), FEATURE_DIM):
        raise ValueError(f"features.npy shape {features.shape} does not match {len(ids)} image IDs.")
    return features, {image_id: i for i, image_id in enumerate(ids)}


def features_for_ids(features: np.ndarray, row_of: dict[str, int], image_ids) -> np.ndarray:
    """Stack the feature vectors for ``image_ids`` (e.g. one split) in that order."""
    return features[[row_of[i] for i in image_ids]]


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Extract VGG16 fc2 features for Flickr8k.")
    parser.add_argument("--data-dir", help="Folder with Images/ and captions.txt (default: data/flickr8k)")
    parser.add_argument("--output-dir", default=str(DEFAULT_FEATURES_DIR))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, help="Only the first N images (smoke test)")
    args = parser.parse_args(argv)

    images_dir, captions_file = get_dataset_paths(args.data_dir)
    ids = sorted(load_captions(captions_file))
    if args.limit:
        ids = ids[:args.limit]
    extract_features(ids, images_dir, args.output_dir, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
