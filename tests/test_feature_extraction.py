"""Tests for src/feature_extraction.py.

* Fast tests use synthetic images and a tiny fake encoder, so storage, alignment
  and resume logic can be checked in seconds.
* ``vgg16`` tests build the real ImageNet VGG16 (weights are downloaded once to
  ~/.keras/models, ~528 MB).
* ``full_features`` tests check data/features after a full extraction and are
  skipped if it has not been run.

Run:  python -m pytest -v tests/test_feature_extraction.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.feature_extraction import (
    DEFAULT_FEATURES_DIR,
    FEATURE_DIM,
    build_encoder,
    extract_features,
    features_for_ids,
    load_features,
    load_image,
    preprocess_batch,
)

IMAGENET_BGR_MEAN = np.array([103.939, 116.779, 123.68], dtype=np.float32)


class FakeEncoder:
    """Stands in for VGG16: a deterministic (batch, 224, 224, 3) -> (batch, 4096) map.

    Each image's mean pixel value is spread over 4,096 outputs, so different
    images give different, predictable vectors. Optionally fails after N batches
    to simulate an interruption.
    """

    def __init__(self, fail_after=None):
        self.calls, self.fail_after, self.rows_seen = 0, fail_after, 0

    def predict_on_batch(self, x):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise KeyboardInterrupt("simulated interruption")
        self.calls += 1
        self.rows_seen += len(x)
        return fake_features(x)


def fake_features(x):
    return (x.mean(axis=(1, 2, 3))[:, None] + np.arange(FEATURE_DIM)[None, :]).astype(np.float32)


@pytest.fixture
def image_dir(tmp_path):
    """Seven PNG images with different solid colours and sizes (one greyscale)."""
    d = tmp_path / "Images"
    d.mkdir()
    for i in range(6):
        Image.new("RGB", (60 + 10 * i, 40), (40 * i, 255 - 30 * i, 17 * i)).save(d / f"img{i}.jpg", "PNG")
    Image.new("L", (50, 50), 128).save(d / "grey.jpg", "PNG")
    return d


IDS = [f"img{i}.jpg" for i in range(6)] + ["grey.jpg"]


# --- Image loading and VGG16 preprocessing ----------------------------------

def test_load_image_resizes_and_converts_to_rgb(image_dir):
    for name in ("img0.jpg", "grey.jpg"):
        img = load_image(image_dir / name)
        assert img.shape == (224, 224, 3) and img.dtype == np.float32
    grey = load_image(image_dir / "grey.jpg")
    assert np.allclose(grey, 128)                       # grey copied to all 3 channels


def test_load_image_keeps_0_255_range(image_dir):
    img = load_image(image_dir / "img0.jpg")             # colour (0, 255, 0)
    assert img.max() == 255 and img.min() == 0           # not rescaled to [0, 1]
    assert np.allclose(img[0, 0], [0, 255, 0])           # RGB order


def test_preprocess_matches_keras_vgg16_caffe_mode():
    rgb = np.zeros((1, 224, 224, 3), np.float32)
    rgb[..., 0] = 255                                    # pure red in RGB
    out = preprocess_batch(rgb)
    # RGB -> BGR, then subtract ImageNet means; red ends up in the LAST channel.
    assert np.allclose(out[0, 0, 0], [0, 0, 255] - IMAGENET_BGR_MEAN, atol=1e-4)
    assert rgb[0, 0, 0, 0] == 255                        # input not modified in place


# --- Storage, alignment and resume (fake encoder) ---------------------------

def test_extract_saves_one_aligned_row_per_image(image_dir, tmp_path):
    out = tmp_path / "features"
    meta = extract_features(IDS + IDS[:3], image_dir, out, batch_size=3,
                            encoder=FakeEncoder(), verbose=False)
    features, row_of = load_features(out)

    assert features.shape == (len(IDS), FEATURE_DIM)     # duplicates removed
    assert sorted(row_of) == sorted(IDS) and len(set(row_of.values())) == len(IDS)
    assert json.loads((out / "image_ids.json").read_text()) == sorted(IDS)
    assert meta["complete"] and meta["num_extracted"] == len(IDS) and not meta["failed_images"]
    for image_id in IDS:                                 # row <-> image ID alignment
        expected = fake_features(preprocess_batch(load_image(image_dir / image_id)[None]))[0]
        assert np.allclose(features[row_of[image_id]], expected)


def test_save_load_round_trip_and_split_lookup(image_dir, tmp_path):
    out = tmp_path / "features"
    extract_features(IDS, image_dir, out, batch_size=4, encoder=FakeEncoder(), verbose=False)
    features, row_of = load_features(out)
    mm, _ = load_features(out, mmap=True)
    assert np.array_equal(features, mm)
    split = ["img3.jpg", "grey.jpg"]
    assert np.array_equal(features_for_ids(features, row_of, split),
                          np.stack([features[row_of["img3.jpg"]], features[row_of["grey.jpg"]]]))


def test_resume_after_interruption_skips_finished_rows(image_dir, tmp_path):
    out = tmp_path / "features"
    with pytest.raises(KeyboardInterrupt):
        extract_features(IDS, image_dir, out, batch_size=2, encoder=FakeEncoder(fail_after=2),
                         verbose=False)
    assert np.load(out / "progress.npy").sum() == 4      # 2 batches x 2 images saved
    with pytest.raises(ValueError, match="incomplete"):
        load_features(out)

    resumed = FakeEncoder()
    extract_features(IDS, image_dir, out, batch_size=2, encoder=resumed, verbose=False)
    assert resumed.rows_seen == len(IDS) - 4             # finished rows not recomputed

    fresh = tmp_path / "fresh"
    extract_features(IDS, image_dir, fresh, batch_size=7, encoder=FakeEncoder(), verbose=False)
    assert np.array_equal(load_features(out)[0], load_features(fresh)[0])   # no corruption


def test_rerun_on_complete_features_changes_nothing(image_dir, tmp_path):
    out = tmp_path / "features"
    extract_features(IDS, image_dir, out, encoder=FakeEncoder(), verbose=False)
    before = load_features(out)[0].copy()
    enc = FakeEncoder()
    extract_features(IDS, image_dir, out, encoder=enc, verbose=False)
    assert enc.calls == 0 and np.array_equal(before, load_features(out)[0])


def test_resume_with_different_image_list_is_refused(image_dir, tmp_path):
    out = tmp_path / "features"
    extract_features(IDS[:3], image_dir, out, encoder=FakeEncoder(), verbose=False)
    with pytest.raises(ValueError, match="different image list"):
        extract_features(IDS, image_dir, out, encoder=FakeEncoder(), verbose=False)


def test_unreadable_image_is_reported_and_retried(image_dir, tmp_path):
    (image_dir / "img2.jpg").write_bytes(b"not an image")
    out = tmp_path / "features"
    meta = extract_features(IDS, image_dir, out, encoder=FakeEncoder(), verbose=False)
    assert list(meta["failed_images"]) == ["img2.jpg"] and not meta["complete"]

    Image.new("RGB", (30, 30), (1, 2, 3)).save(image_dir / "img2.jpg", "PNG")   # fix the file
    meta = extract_features(IDS, image_dir, out, encoder=FakeEncoder(), verbose=False)
    assert meta["complete"] and not meta["failed_images"]


def test_load_features_missing_dir_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.feature_extraction"):
        load_features(tmp_path / "nothing")


# --- Real VGG16 encoder ------------------------------------------------------

@pytest.fixture(scope="module")
def encoder():
    return build_encoder()


def test_encoder_input_output_shapes(encoder):
    assert encoder.input_shape == (None, 224, 224, 3)
    assert encoder.output_shape == (None, 4096)
    assert encoder.layers[-1].name == "fc2"


def test_imagenet_classification_layer_removed(encoder):
    names = [layer.name for layer in encoder.layers]
    assert "predictions" not in names and "fc1" in names and "fc2" in names


def test_encoder_is_frozen(encoder):
    assert not encoder.trainable
    assert all(not layer.trainable for layer in encoder.layers)
    assert encoder.trainable_weights == []
    assert len(encoder.non_trainable_weights) == 30      # 13 conv + 2 dense layers, kernel + bias


def test_encoder_uses_imagenet_weights(encoder):
    import h5py
    path = Path(os.path.expanduser("~/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5"))
    with h5py.File(path, "r") as f:                      # the official Keras ImageNet file
        assert np.array_equal(encoder.get_layer("fc2").kernel.numpy(), f["fc2"]["fc2_W_1:0"][()])
        assert np.array_equal(encoder.get_layer("block1_conv1").kernel.numpy(),
                              f["block1_conv1"]["block1_conv1_W_1:0"][()])


def test_real_encoder_batch_output_is_finite_and_batch_independent(encoder, image_dir):
    x = preprocess_batch(np.stack([load_image(image_dir / i) for i in IDS[:4]]))
    batch = np.asarray(encoder.predict_on_batch(x))
    assert batch.shape == (4, 4096) and np.isfinite(batch).all()
    single = np.asarray(encoder.predict_on_batch(x[2:3]))[0]
    assert np.allclose(batch[2], single, atol=1e-4)      # batching does not mix images


# --- Full extraction results (skipped if not run yet) -------------------------

def _features_ready():
    try:
        load_features(DEFAULT_FEATURES_DIR, mmap=True)
        return True
    except (FileNotFoundError, ValueError):
        return False


full_features = pytest.mark.skipif(not _features_ready(), reason="Run python -m src.feature_extraction first")


@full_features
def test_full_features_cover_every_image_exactly_once():
    from src.data_preprocessing import get_dataset_paths, load_captions
    features, row_of = load_features(mmap=True)
    caption_ids = set(load_captions(get_dataset_paths()[1]))
    assert features.shape == (8091, 4096) and features.dtype == np.float32
    assert set(row_of) == caption_ids and len(row_of) == features.shape[0]
    assert np.isfinite(features).all()


@full_features
def test_full_features_match_saved_splits():
    features, row_of = load_features(mmap=True)
    splits = json.loads((DEFAULT_FEATURES_DIR.parent / "processed" / "splits.json").read_text())
    for name, ids in splits.items():
        assert features_for_ids(features, row_of, ids).shape == (len(ids), 4096), name
    assert sum(len(v) for v in splits.values()) == len(row_of)


@full_features
def test_full_features_spot_check_against_fresh_extraction(encoder):
    from src.data_preprocessing import get_dataset_paths
    features, row_of = load_features(mmap=True)
    images_dir = get_dataset_paths()[0]
    for image_id in ("1000268201_693b08cb0e.jpg", sorted(row_of)[-1]):
        x = preprocess_batch(load_image(images_dir / image_id)[None])
        assert np.allclose(features[row_of[image_id]], encoder.predict_on_batch(x)[0], atol=1e-4)
