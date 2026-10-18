"""Tests for greedy and standard beam search; do not need Flickr8k or a GPU."""
import numpy as np
import pytest

from src.caption_decoding import beam_search_decode, greedy_decode
from src.feature_extraction import FEATURE_DIM


class TinyTokenizer:
    word_index = {"<unk>": 1, "startseq": 2, "endseq": 3, "cat": 4, "dog": 5}
    index_word = {v: k for k, v in word_index.items()}


class DummyModel:
    """Greedy prefers 'cat' but beam search finds better full caption 'dog'.

    startseq -> cat .60 or dog .40
    cat      -> endseq .51 or dog .49
    dog      -> endseq .99 or cat .01

    Greedy scores cat endseq: 0.6*.51 = .306
    Beam scores dog endseq:   0.4*.99 = .396 (globally more likely)
    """

    def __init__(self):
        self.batch_sizes = []

    def predict_on_batch(self, inputs):
        captions = inputs["caption_input"]
        features = inputs["image_features"]
        assert captions.shape[0] == features.shape[0]
        assert features.shape[1] == FEATURE_DIM
        self.batch_sizes.append(len(captions))
        output = np.zeros((len(captions), 6), dtype=np.float64)
        for row, caption in enumerate(captions):
            non_padding = [int(i) for i in caption if int(i) != 0]
            if non_padding == [2]:
                output[row, 4] = .60
                output[row, 5] = .40
            elif non_padding[-1] == 4:
                output[row, 3] = .51
                output[row, 5] = .49
            else:  # dog -> endseq
                output[row, 3] = .99
                output[row, 4] = .01
        return output


def test_greedy_and_beam_differ_when_local_choice_is_suboptimal():
    model = DummyModel()
    vector = np.zeros(FEATURE_DIM, dtype=np.float32)
    tok = TinyTokenizer()

    assert greedy_decode(model, vector, tok, max_length=6) == "cat"
    assert beam_search_decode(model, vector, tok, max_length=6, beam_width=3) == "dog"
    assert beam_search_decode(model, vector, tok, max_length=6, beam_width=5) == "dog"


def test_beam_width_one_matches_greedy():
    model = DummyModel()
    features = np.ones(FEATURE_DIM, dtype=np.float32)
    tokenizer = TinyTokenizer()
    greedy = greedy_decode(model, features, tokenizer, max_length=6)
    beam_one = beam_search_decode(model, features, tokenizer, max_length=6, beam_width=1)
    assert greedy == beam_one


def test_beam_search_batches_active_hypotheses():
    model = DummyModel()
    beam_search_decode(model, np.ones(FEATURE_DIM), TinyTokenizer(), max_length=6, beam_width=3)
    assert max(model.batch_sizes) > 1


@pytest.mark.parametrize("bad_shape", [(224, 224, 3), (1, FEATURE_DIM), (10,)])
def test_decoder_rejects_incorrect_feature_dimensions(bad_shape):
    with pytest.raises(ValueError, match="Expected VGG16"):
        greedy_decode(DummyModel(), np.zeros(bad_shape), TinyTokenizer(), max_length=5)


def test_decoder_rejects_invalid_beam_width():
    with pytest.raises(ValueError, match="beam_width"):
        beam_search_decode(DummyModel(), np.zeros(FEATURE_DIM), TinyTokenizer(),
                           max_length=5, beam_width=0)


def test_decoder_rejects_invalid_max_length():
    with pytest.raises(ValueError, match="max_length"):
        greedy_decode(DummyModel(), np.zeros(FEATURE_DIM), TinyTokenizer(), max_length=1)
