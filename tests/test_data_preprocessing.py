"""Tests for src/data_preprocessing.py.

Fixture tests use a tiny fake Flickr8k folder and always run.
Tests marked ``full_dataset`` use the real data in data/flickr8k (or FLICKR8K_DIR)
and are skipped automatically if it is not present.

Run:  python -m pytest -v
"""

import numpy as np
import pytest

from src.data_preprocessing import (
    END_TOKEN,
    OOV_TOKEN,
    START_TOKEN,
    CaptionTokenizer,
    clean_caption,
    create_sequences,
    get_dataset_paths,
    get_max_length,
    load_captions,
    prepare_flickr8k,
    split_image_ids,
    verify_image_files,
)

# Same format as the real captions.txt, including a quoted caption with a comma.
FIXTURE_CSV = """image,caption
dog.jpg,A dog runs .
dog.jpg,"A brown dog , running fast ."
cat.jpg,A cat sleeps on a mat .
cat.jpg,The cat 's tail is tri-colored !
"""


@pytest.fixture
def fake_dataset(tmp_path):
    """A minimal Flickr8k-style folder: Images/ + captions.txt."""
    (tmp_path / "Images").mkdir()
    for name in ("dog.jpg", "cat.jpg"):
        (tmp_path / "Images" / name).write_bytes(b"")
    (tmp_path / "captions.txt").write_text(FIXTURE_CSV, encoding="utf-8")
    return tmp_path


# --- 1-2. Parsing and image -> caption mapping ------------------------------

def test_load_captions_parses_quoted_commas_and_groups_by_image(fake_dataset):
    caps = load_captions(fake_dataset / "captions.txt")
    assert caps == {
        "dog.jpg": ["A dog runs .", "A brown dog , running fast ."],
        "cat.jpg": ["A cat sleeps on a mat .", "The cat 's tail is tri-colored !"],
    }


def test_load_captions_rejects_wrong_header(tmp_path):
    f = tmp_path / "captions.txt"
    f.write_text("filename\tcaption\nx.jpg\tA dog\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected header"):
        load_captions(f)


def test_load_captions_rejects_malformed_row(tmp_path):
    f = tmp_path / "captions.txt"
    f.write_text("image,caption\nx.jpg,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_captions(f)


# --- 3. Useful errors for invalid paths -------------------------------------

def test_missing_dataset_folder_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="kaggle.com"):
        get_dataset_paths(tmp_path / "does_not_exist")


def test_missing_captions_file(fake_dataset):
    (fake_dataset / "captions.txt").unlink()
    with pytest.raises(FileNotFoundError, match="Caption file not found"):
        get_dataset_paths(fake_dataset)


def test_missing_images_folder(tmp_path):
    (tmp_path / "captions.txt").write_text(FIXTURE_CSV, encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Image folder not found"):
        get_dataset_paths(tmp_path)


def test_missing_image_file_is_reported(fake_dataset):
    (fake_dataset / "Images" / "cat.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="cat.jpg"):
        verify_image_files(["dog.jpg", "cat.jpg"], fake_dataset / "Images")


def test_data_dir_can_come_from_environment(fake_dataset, monkeypatch):
    monkeypatch.setenv("FLICKR8K_DIR", str(fake_dataset))
    images_dir, captions_file = get_dataset_paths()
    assert images_dir == fake_dataset / "Images"
    assert captions_file == fake_dataset / "captions.txt"


# --- 8. Cleaning and special tokens -----------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("A dog runs .", "startseq a dog runs endseq"),
    ("A brown dog , running fast .", "startseq a brown dog running fast endseq"),
    ("The cat 's tail is tri-colored !", "startseq the cat tail is tri colored endseq"),
    ("Two   dogs  ISN'T here", "startseq two dogs isnt here endseq"),
    ("3 kids on a slide", "startseq 3 kids on a slide endseq"),
])
def test_clean_caption(raw, expected):
    assert clean_caption(raw) == expected


def test_special_tokens_appear_exactly_once_at_the_ends():
    words = clean_caption("A dog jumps over a log .").split()
    assert words[0] == START_TOKEN and words[-1] == END_TOKEN
    assert words.count(START_TOKEN) == 1 and words.count(END_TOKEN) == 1


# --- 4-5. Splitting ---------------------------------------------------------

IDS = [f"img_{i:04d}.jpg" for i in range(1000)]


def test_split_sizes_and_no_overlap():
    s = split_image_ids(IDS)
    train, val, test = set(s["train"]), set(s["val"]), set(s["test"])
    assert (len(train), len(val), len(test)) == (800, 100, 100)
    assert not (train & val) and not (train & test) and not (val & test)
    assert train | val | test == set(IDS)


def test_split_is_reproducible_and_independent_of_input_order():
    a = split_image_ids(IDS, seed=42)
    b = split_image_ids(list(reversed(IDS)), seed=42)
    assert a == b
    assert split_image_ids(IDS, seed=7) != a


def test_split_keeps_duplicate_ids_together():
    # captions.txt lists each image 5 times; duplicates must collapse to one ID.
    s = split_image_ids([i for i in IDS for _ in range(5)])
    assert sum(len(v) for v in s.values()) == len(IDS)


# --- 6-7. Tokenizer: training-only vocabulary, unknown words ----------------

def test_tokenizer_uses_training_captions_only_and_maps_unknown_words():
    train = ["startseq a dog runs endseq", "startseq a dog sits endseq"]
    tok = CaptionTokenizer().fit(train)

    assert "zebra" not in tok.word_index
    assert tok.word_index[OOV_TOKEN] == 1
    assert 0 not in tok.index_word                      # 0 is padding only
    assert tok.vocab_size == 7 + 1                      # 6 words + <unk> + padding

    seq = tok.texts_to_sequences(["startseq a zebra runs endseq"])[0]
    assert seq[2] == 1                                  # unseen word -> <unk>
    assert tok.sequences_to_texts([seq]) == ["startseq a <unk> runs endseq"]


def test_tokenizer_is_deterministic_and_round_trips(tmp_path):
    texts = ["startseq b a endseq", "startseq a b endseq", "startseq c endseq"]
    tok = CaptionTokenizer().fit(texts)
    assert tok.word_index == CaptionTokenizer().fit(reversed(texts)).word_index
    tok.save(tmp_path / "tok.json")
    assert CaptionTokenizer.load(tmp_path / "tok.json").word_index == tok.word_index


def test_min_count_sends_rare_words_to_unk():
    tok = CaptionTokenizer(min_count=2).fit(["startseq a dog endseq", "startseq a cat endseq"])
    assert "dog" not in tok.word_index and "a" in tok.word_index


# --- 9-10. Next-word sequences: alignment and shapes ------------------------

def test_create_sequences_input_target_alignment():
    caption = "startseq a dog runs endseq"
    tok = CaptionTokenizer().fit([caption])
    ids, X, y = create_sequences({"dog.jpg": [caption]}, tok, max_length=5)

    assert X.shape == (4, 5) and y.shape == (4,) and ids.shape == (4,)
    assert X.dtype == np.int32 and y.dtype == np.int32
    assert list(ids) == ["dog.jpg"] * 4

    # Remove left padding and decode, to compare with the expected table.
    inputs = tok.sequences_to_texts(X)
    targets = [tok.index_word[int(t)] for t in y]
    assert list(zip(inputs, targets)) == [
        ("startseq", "a"),
        ("startseq a", "dog"),
        ("startseq a dog", "runs"),
        ("startseq a dog runs", "endseq"),
    ]
    assert list(X[0]) == [0, 0, 0, 0, tok.word_index["startseq"]]   # pre-padding


def test_create_sequences_pair_count_and_truncation():
    tok = CaptionTokenizer().fit(["startseq a b c d e endseq"])
    caps = {"x.jpg": ["startseq a b c d e endseq"], "y.jpg": ["startseq a endseq"]}
    ids, X, y = create_sequences(caps, tok, max_length=3)
    assert len(y) == (7 - 1) + (3 - 1)                  # one pair per word after startseq
    assert X.shape == (8, 3)
    # Longest prefix (6 words) is truncated to its last 3 words.
    assert tok.sequences_to_texts([X[5]]) == ["c d e"]


def test_get_max_length_counts_special_tokens(fake_dataset):
    from src.data_preprocessing import clean_captions
    caps = clean_captions(load_captions(fake_dataset / "captions.txt"))
    assert get_max_length(caps) == len("startseq the cat tail is tri colored endseq".split())


def test_prepare_on_fixture_fits_tokenizer_on_train_only(fake_dataset):
    data = prepare_flickr8k(fake_dataset)
    train_words = {w for caps in data.split_captions("train").values() for c in caps for w in c.split()}
    assert set(data.tokenizer.word_index) == train_words | {OOV_TOKEN}


# --- Full dataset checks (skipped if Flickr8k is not downloaded) -------------

def _dataset_available():
    try:
        get_dataset_paths()
        return True
    except FileNotFoundError:
        return False


full_dataset = pytest.mark.skipif(not _dataset_available(), reason="Flickr8k not found in data/flickr8k")


@pytest.fixture(scope="module")
def full():
    return prepare_flickr8k()


@full_dataset
def test_full_dataset_counts(full):
    assert len(full.captions) == 8091
    assert sum(len(c) for c in full.captions.values()) == 40455
    assert all(len(c) == 5 for c in full.captions.values())


@full_dataset
def test_full_dataset_known_image_captions(full):
    caps = full.captions["1000268201_693b08cb0e.jpg"]
    assert caps[0] == "startseq a child in a pink dress is climbing up a set of stairs in an entry way endseq"
    assert len(caps) == 5


@full_dataset
def test_full_dataset_split_has_no_leakage(full):
    s = {k: set(v) for k, v in full.splits.items()}
    assert (len(s["train"]), len(s["val"]), len(s["test"])) == (6472, 809, 810)
    assert not (s["train"] & s["val"]) and not (s["train"] & s["test"]) and not (s["val"] & s["test"])
    assert s["train"] | s["val"] | s["test"] == set(full.captions)
    assert prepare_flickr8k().splits == full.splits     # repeated run -> identical split


@full_dataset
def test_full_dataset_vocabulary_is_train_only(full):
    train_words = {w for caps in full.split_captions("train").values() for c in caps for w in c.split()}
    assert set(full.tokenizer.word_index) == train_words | {OOV_TOKEN}
