import pytest

from zerogpt.tokenizer import Tokenizer


@pytest.fixture
def trained_tokenizer():
    tok = Tokenizer()
    tok.train(["hello", "world"])
    return tok


def test_vocab_size_before_training_raises():
    # GIVEN a freshly created (untrained) tokenizer
    tok = Tokenizer()

    # WHEN reading `vocab_size`
    # THEN a `RuntimeError` is raised
    with pytest.raises(RuntimeError, match="trained"):
        _ = tok.vocab_size


def test_encode_before_training_raises():
    # GIVEN a freshly created (untrained) tokenizer
    tok = Tokenizer()

    # WHEN attempting to encode a string
    # THEN a `RuntimeError` is raised
    with pytest.raises(RuntimeError, match="trained"):
        tok.encode("hi")


def test_decode_before_training_raises():
    # GIVEN a freshly created (untrained) tokenizer
    tok = Tokenizer()

    # WHEN attempting to decode token ids
    # THEN a `RuntimeError` is raised
    with pytest.raises(RuntimeError, match="trained"):
        tok.decode([0, 1])


def test_bos_token_before_training_raises():
    # GIVEN a freshly created (untrained) tokenizer
    tok = Tokenizer()

    # WHEN reading `bos_token`
    # THEN a `RuntimeError` is raised
    with pytest.raises(RuntimeError, match="trained"):
        _ = tok.bos_token


def test_train_builds_vocab_from_unique_characters():
    # GIVEN a corpus containing the three unique characters {a, b, c}
    tok = Tokenizer()

    # WHEN training the tokenizer
    tok.train(["ba", "cba"])

    # THEN `vocab_size` is the count of unique chars (3) plus the 3 special tokens
    assert tok.vocab_size == 3 + 3


def test_special_tokens_are_distinct(trained_tokenizer):
    # GIVEN a trained tokenizer
    # WHEN collecting the three special token ids
    bos = trained_tokenizer.bos_token
    eos = trained_tokenizer.eos_token
    unk = trained_tokenizer.unk_token

    # THEN they are all distinct
    assert len({bos, eos, unk}) == 3


def test_special_tokens_occupy_top_of_index_range(trained_tokenizer):
    # GIVEN a trained tokenizer
    # WHEN looking at the special token ids
    specials = trained_tokenizer.special_tokens

    # THEN they occupy the highest indices in the vocabulary
    assert max(specials) == trained_tokenizer.vocab_size - 1


def test_encode_wraps_with_bos_and_eos(trained_tokenizer):
    # GIVEN a trained tokenizer
    # WHEN encoding a string
    tokens = trained_tokenizer.encode("hello")

    # THEN the token sequence begins with `bos` and ends with `eos`
    assert tokens[0] == trained_tokenizer.bos_token
    assert tokens[-1] == trained_tokenizer.eos_token


def test_encode_uses_unk_for_unknown_characters(trained_tokenizer):
    # GIVEN a tokenizer trained on `hello world` (which does not contain `z`)
    # WHEN encoding the single character `z`
    tokens = trained_tokenizer.encode("z")

    # THEN the middle token (between bos and eos) is `unk`
    assert tokens[1] == trained_tokenizer.unk_token


def test_decode_strips_special_tokens(trained_tokenizer):
    # GIVEN a string encoded with the tokenizer (so it's wrapped in bos/eos)
    encoded = trained_tokenizer.encode("hello")

    # WHEN decoding it back to a string
    decoded = trained_tokenizer.decode(encoded)

    # THEN the bos/eos wrappers are stripped, leaving the original text
    assert decoded == "hello"


def test_decode_drops_unknown_characters(trained_tokenizer):
    # GIVEN a document with one character outside the vocab (encoded as `unk`)
    encoded = trained_tokenizer.encode("hez")

    # WHEN decoding it back to a string
    decoded = trained_tokenizer.decode(encoded)

    # THEN the unknown character is stripped along with bos/eos
    assert decoded == "he"


def test_special_tokens_property_returns_all_three(trained_tokenizer):
    # GIVEN a trained tokenizer
    # WHEN reading the `special_tokens` property
    specials = trained_tokenizer.special_tokens

    # THEN it contains bos, eos, and unk
    assert len(specials) == 3
    assert trained_tokenizer.bos_token in specials
    assert trained_tokenizer.eos_token in specials
    assert trained_tokenizer.unk_token in specials


def test_retraining_overwrites_previous_vocab():
    # GIVEN a tokenizer that was trained on one corpus
    tok = Tokenizer()
    tok.train(["abc"])
    first_size = tok.vocab_size

    # WHEN training again on a different corpus
    tok.train(["wxyz"])

    # THEN the vocab is replaced (not extended): 4 unique chars + 3 special tokens
    assert tok.vocab_size == 4 + 3
    assert tok.vocab_size != first_size
