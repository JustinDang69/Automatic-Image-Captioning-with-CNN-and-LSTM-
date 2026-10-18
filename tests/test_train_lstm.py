"""Tests for src/train_lstm.py.

* Model / batching tests use synthetic data (seconds, CPU only).
* ``train()`` tests use a tiny fake Flickr8k folder and random features.
* ``real_data`` test trains 1 epoch on 10 real images; skipped if data/features
  are missing.

Run:  python -m pytest -v tests/test_train_lstm.py
"""

import json
import math

import keras
import numpy as np
import pytest

from src.feature_extraction import FEATURE_DIM
from src.train_lstm import (
    BEST_MODEL_FILE,
    FINAL_MODEL_FILE,
    LAST_MODEL_FILE,
    STATE_FILE,
    CaptionBatches,
    ResumableTraining,
    build_caption_model,
    load_training_state,
    prepare_batches,
    train,
)

VOCAB, MAXLEN = 50, 8


@pytest.fixture(scope="module")
def model():
    return build_caption_model(VOCAB, MAXLEN)


def synthetic_batch(n=16, seed=0):
    rng = np.random.default_rng(seed)
    image = rng.random((n, FEATURE_DIM), dtype=np.float32)
    caption = np.zeros((n, MAXLEN), np.int32)
    for i in range(n):                                   # left-padded like create_sequences
        length = rng.integers(1, MAXLEN + 1)
        caption[i, MAXLEN - length:] = rng.integers(2, VOCAB, length)
    target = rng.integers(1, VOCAB, n).astype(np.int32)
    return {"image_features": image, "caption_input": caption}, target


# --- Model structure ----------------------------------------------------------

def test_model_inputs_outputs_and_fusion_shapes(model):
    assert model.input_shape == [(None, FEATURE_DIM), (None, MAXLEN)]
    assert model.output_shape == (None, VOCAB)
    shapes = {l.name: l.output.shape for l in model.layers}
    assert shapes["image_dense_256"] == (None, 256)
    assert shapes["embedding_256"] == (None, MAXLEN, 256)
    assert shapes["lstm_256"] == (None, 256)
    assert shapes["feature_fusion"] == (None, 256)
    assert shapes["fusion_dense_256"] == (None, 256)
    assert isinstance(model.get_layer("feature_fusion"), keras.layers.Add)   # element-wise addition


def test_only_decoder_weights_are_trained(model):
    # VGG16 is not part of this model: every parameter belongs to the decoder.
    expected = (
        FEATURE_DIM * 256 + 256                   # image Dense(256)
        + VOCAB * 256                             # Embedding(256)
        + 4 * (256 * 256 + 256 * 256 + 256)       # LSTM(256)
        + 256 * 256 + 256                         # fusion Dense(256)
        + 256 * VOCAB + VOCAB                     # softmax
    )
    assert model.count_params() == expected
    assert not any("block" in l.name or l.name in ("fc1", "fc2") for l in model.layers)


def test_compiled_for_sparse_integer_targets(model):
    assert model.loss == "sparse_categorical_crossentropy"
    assert isinstance(model.optimizer, keras.optimizers.Adam)


def test_default_model_is_cudnn_compatible_with_left_padding():
    m = build_caption_model(VOCAB, MAXLEN)
    assert m.get_layer("embedding_256").mask_zero is False   # no left-padded mask reaches cuDNN
    assert m.get_layer("lstm_256").use_cudnn == "auto"
    masked = build_caption_model(VOCAB, MAXLEN, mask_padding=True)
    assert masked.get_layer("embedding_256").mask_zero is True
    assert masked.get_layer("lstm_256").use_cudnn is False   # cuDNN would reject this mask


def test_why_left_padded_masks_cannot_use_cudnn():
    """Documents the Keras 3 rule behind mask_padding (uses a private Keras helper)."""
    import importlib
    import tensorflow as tf
    rnn = importlib.import_module("keras.src.backend.tensorflow.rnn")
    left = tf.constant([[False, False, True, True]])     # our pre-padded captions
    right = tf.constant([[True, True, False, False]])
    assert not bool(rnn._is_sequence_right_padded(left))
    assert bool(rnn._is_sequence_right_padded(right))


@pytest.mark.parametrize("mask_padding", [True, False])
def test_effect_of_masking_on_left_padding(mask_padding):
    """[0,0,0,0,a,b,c,d] vs [a,b,c,d] with identical weights.

    Masked: padding is skipped, so both give the same prediction.
    Unmasked (default): the LSTM also reads the 4 padding steps, so they differ.
    """
    long = build_caption_model(VOCAB, 8, mask_padding=mask_padding, dropout=0.0)
    short = build_caption_model(VOCAB, 4, mask_padding=mask_padding, dropout=0.0)
    short.set_weights(long.get_weights())     # weights do not depend on max_length
    image = np.random.default_rng(1).random((1, FEATURE_DIM), dtype=np.float32)
    words = np.array([[7, 3, 9, 4]], np.int32)
    padded = np.concatenate([np.zeros((1, 4), np.int32), words], axis=1)
    a = long.predict_on_batch({"image_features": image, "caption_input": padded})
    b = short.predict_on_batch({"image_features": image, "caption_input": words})
    assert np.allclose(a, b, atol=1e-5) == mask_padding


# --- Forward pass and single-batch training -----------------------------------

def test_forward_pass_gives_probability_distribution(model):
    x, _ = synthetic_batch(4)
    probs = np.asarray(model.predict_on_batch(x))
    assert probs.shape == (4, VOCAB)
    assert np.isfinite(probs).all() and np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_single_batch_training_reduces_loss():
    keras.utils.set_random_seed(0)
    m = build_caption_model(VOCAB, MAXLEN, dropout=0.0)
    x, y = synthetic_batch(16)
    first = float(m.train_on_batch(x, y, return_dict=True)["loss"])
    for _ in range(30):
        last = float(m.train_on_batch(x, y, return_dict=True)["loss"])
    assert math.isfinite(first) and last < first * 0.5          # it can learn (overfit one batch)


# --- Batching -------------------------------------------------------------------

def test_caption_batches_align_features_prefixes_and_targets():
    features = np.arange(5 * FEATURE_DIM, dtype=np.float32).reshape(5, FEATURE_DIM)
    rows = np.array([4, 0, 2, 2, 1, 3, 0])
    prefixes = np.arange(7 * MAXLEN).reshape(7, MAXLEN)
    targets = np.arange(10, 17)
    ds = CaptionBatches(features, rows, prefixes, targets, batch_size=3, shuffle=True, seed=1)

    assert len(ds) == 3                                       # ceil(7 / 3)
    seen = []
    for b in range(len(ds)):
        x, y = ds[b]
        assert x["image_features"].shape[1] == FEATURE_DIM and x["caption_input"].shape[1] == MAXLEN
        for img, pre, t in zip(x["image_features"], x["caption_input"], y):
            k = t - 10                                        # recover the original pair index
            assert np.array_equal(img, features[rows[k]]) and np.array_equal(pre, prefixes[k])
            seen.append(k)
    assert sorted(seen) == list(range(7))                     # every pair exactly once per epoch


def test_validation_batches_are_not_shuffled_and_misalignment_is_rejected():
    ds = CaptionBatches(np.zeros((1, FEATURE_DIM)), [0] * 5, np.zeros((5, MAXLEN)), np.arange(5),
                        batch_size=2, shuffle=False)
    assert list(ds.order) == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="align"):
        CaptionBatches(np.zeros((1, FEATURE_DIM)), [0] * 4, np.zeros((5, MAXLEN)), np.arange(5))


# --- Resumable training callback ---------------------------------------------

def _tiny_compiled_model():
    m = keras.Sequential([keras.Input((2,)), keras.layers.Dense(2, activation="softmax")])
    m.compile(keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy")
    return m


def test_callback_early_stopping_lr_reduction_and_state_survive_restart(tmp_path):
    m = _tiny_compiled_model()
    cb = ResumableTraining(tmp_path, config={}, patience=4, lr_patience=2)
    cb.set_model(m)
    for epoch, val in enumerate([1.0, 1.1, 1.2]):            # best at epoch 1, then 2 bad epochs
        cb.on_epoch_end(epoch, {"loss": 1.0, "val_loss": val})
    assert cb.best_epoch == 1 and cb.wait == 2
    assert math.isclose(float(m.optimizer.learning_rate), 5e-4, rel_tol=1e-5)   # halved
    assert not m.stop_training

    # "Colab restarts": rebuild everything from the files on disk.
    state = load_training_state(tmp_path)
    assert state["epochs_completed"] == 3 and state["wait"] == 2
    m2 = keras.models.load_model(tmp_path / LAST_MODEL_FILE)
    assert math.isclose(float(m2.optimizer.learning_rate), 5e-4, rel_tol=1e-5)
    cb2 = ResumableTraining(tmp_path, config={}, patience=4, lr_patience=2, state=state)
    cb2.set_model(m2)
    cb2.on_epoch_end(3, {"loss": 1.0, "val_loss": 1.3})
    assert not m2.stop_training                               # 3 bad epochs < patience 4
    cb2.on_epoch_end(4, {"loss": 1.0, "val_loss": 1.4})
    assert m2.stop_training and cb2.stopped_early             # 4th bad epoch -> stop
    assert len(cb2.history["val_loss"]) == 5
    assert (tmp_path / BEST_MODEL_FILE).is_file() and (tmp_path / "training_loss.png").is_file()


def test_callback_rejects_nan_validation_loss(tmp_path):
    cb = ResumableTraining(tmp_path, config={})
    cb.set_model(_tiny_compiled_model())
    with pytest.raises(RuntimeError, match="nan"):
        cb.on_epoch_end(0, {"loss": 1.0, "val_loss": float("nan")})


# --- train() end to end on a tiny fake dataset --------------------------------

@pytest.fixture
def tiny_project(tmp_path):
    """10 fake images with captions + matching random 'VGG16' features."""
    data = tmp_path / "flickr8k"
    (data / "Images").mkdir(parents=True)
    words = ["dog", "cat", "runs", "sits", "red", "ball", "grass", "water"]
    lines = ["image,caption"]
    ids = [f"img{i}.jpg" for i in range(10)]
    for i, image_id in enumerate(ids):
        (data / "Images" / image_id).write_bytes(b"")
        for j in range(5):
            lines.append(f"{image_id},A {words[(i + j) % 8]} {words[(i * j) % 8]} near the {words[j]} .")
    (data / "captions.txt").write_text("\n".join(lines), encoding="utf-8")

    feats = tmp_path / "features"
    feats.mkdir()
    np.save(feats / "features.npy", np.random.default_rng(0).random((10, FEATURE_DIM), dtype=np.float32))
    (feats / "image_ids.json").write_text(json.dumps(sorted(ids)))
    np.save(feats / "progress.npy", np.ones(10, bool))
    return data, feats, tmp_path / "run"


def test_train_saves_all_artifacts_and_resumes(tiny_project):
    data, feats, out = tiny_project
    kwargs = dict(data_dir=data, features_dir=feats, output_dir=out, batch_size=16, verbose=0)

    _, history, prepared = train(epochs=1, **kwargs)          # session 1 "disconnects" after 1 epoch
    assert len(history["loss"]) == 1
    for name in (BEST_MODEL_FILE, LAST_MODEL_FILE, FINAL_MODEL_FILE, STATE_FILE, "tokenizer.json",
                 "splits.json", "metadata.json", "history.json", "training_history.csv",
                 "training_loss.png", "training_config.json"):
        assert (out / name).is_file(), name
    iters_after_1 = int(keras.models.load_model(out / LAST_MODEL_FILE).optimizer.iterations)

    model, history, _ = train(epochs=3, **kwargs)             # session 2 continues
    assert len(history["loss"]) == 3
    assert load_training_state(out)["epochs_completed"] == 3
    iters_after_3 = int(keras.models.load_model(out / LAST_MODEL_FILE).optimizer.iterations)
    assert iters_after_3 == 3 * iters_after_1                 # weights/optimizer continued, not restarted
    assert len((out / "training_history.csv").read_text().strip().splitlines()) == 1 + 3
    config = json.loads((out / "training_config.json").read_text())
    assert config["epochs_run"] == 3 and config["best_epoch"] >= 1
    assert model.output_shape == (None, prepared.vocab_size)


def test_train_refuses_to_resume_with_changed_settings(tiny_project):
    data, feats, out = tiny_project
    train(data_dir=data, features_dir=feats, output_dir=out, epochs=1, verbose=0)
    with pytest.raises(ValueError, match="dropout"):
        train(data_dir=data, features_dir=feats, output_dir=out, epochs=2, dropout=0.2, verbose=0)
    _, history, _ = train(data_dir=data, features_dir=feats, output_dir=out, epochs=1,
                          dropout=0.2, resume=False, verbose=0)   # explicit fresh start is allowed
    assert len(history["loss"]) == 1


def test_prepare_batches_never_uses_test_split(tiny_project):
    from src.data_preprocessing import prepare_flickr8k
    from src.feature_extraction import load_features
    data, feats, _ = tiny_project
    prepared = prepare_flickr8k(data)
    features, row_of = load_features(feats)
    with pytest.raises(ValueError, match="train and val"):
        prepare_batches(prepared, features, row_of, "test")
    train_ds, _ = prepare_batches(prepared, features, row_of, "train")
    train_rows = {row_of[i] for i in prepared.splits["train"]}
    assert set(np.unique(train_ds.rows)) <= train_rows


# --- Small real-data run (skipped without Flickr8k + features) -----------------

def _real_data_ready():
    from src.data_preprocessing import get_dataset_paths
    from src.feature_extraction import load_features
    try:
        get_dataset_paths()
        load_features(mmap=True)
        return True
    except (FileNotFoundError, ValueError):
        return False


@pytest.mark.skipif(not _real_data_ready(), reason="needs data/flickr8k and data/features")
def test_one_epoch_on_ten_real_images(tmp_path):
    model, history, prepared = train(output_dir=tmp_path, epochs=1, max_images=10, verbose=0)
    assert prepared.vocab_size == 7702 and prepared.max_length == 39   # full training vocabulary
    assert model.input_shape == [(None, 4096), (None, 39)] and model.output_shape == (None, 7702)
    assert math.isfinite(history["val_loss"][0])
