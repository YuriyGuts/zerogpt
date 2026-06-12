import pytest

from tests.helpers import maybe_import

GPTParams = maybe_import("zerogpt.model", "GPTParams")
load_model = maybe_import("zerogpt.serialization", "load_model")
save_model = maybe_import("zerogpt.serialization", "save_model")
Tokenizer = maybe_import("zerogpt.tokenizer", "Tokenizer")


@pytest.fixture
def tokenizer():
    tok = Tokenizer()
    tok.train(["abc", "def"])
    return tok


@pytest.fixture
def params():
    return GPTParams.create(
        vocab_size=10,
        embedding_dim=4,
        max_sequence_length=4,
        transformer_block_count=1,
        attn_head_count=2,
    )


@pytest.fixture
def temp_data_dir(tmp_path, monkeypatch):
    # Redirect `save_model`'s output away from the real `data/` folder.
    monkeypatch.setattr("zerogpt.serialization.data_dir", tmp_path)
    return tmp_path


def test_save_model_writes_file_into_data_dir(params, tokenizer, temp_data_dir):
    # GIVEN params, tokenizer, and a redirected `data_dir`
    # WHEN saving the model
    saved_path = save_model(params, tokenizer)

    # THEN a file exists in the redirected directory
    assert saved_path.exists()
    assert saved_path.parent == temp_data_dir


def test_save_model_filename_encodes_hyperparameters(params, tokenizer, temp_data_dir):
    # GIVEN params with specific vocab/emb/seq/transf/attn values
    # WHEN saving the model
    saved_path = save_model(params, tokenizer)

    # THEN the filename embeds each hyperparameter as `name-value` segments
    name = saved_path.name
    assert f"vocab-{params.vocab_size}" in name
    assert "emb-4" in name
    assert "seq-4" in name
    assert "transf-1" in name
    assert "attn-2" in name
    assert name.endswith(".model")


def test_save_model_filename_includes_extra_id(params, tokenizer, temp_data_dir):
    # GIVEN an `extra_id` representing a checkpoint identifier
    # WHEN saving the model
    saved_path = save_model(params, tokenizer, extra_id="iter-001")

    # THEN the filename contains the extra id segment
    assert "iter-001" in saved_path.name


def test_save_model_filename_omits_extra_id_when_empty(params, tokenizer, temp_data_dir):
    # GIVEN no `extra_id` argument
    # WHEN saving the model
    saved_path = save_model(params, tokenizer)

    # THEN no trailing `-` separator appears before the `.model` extension
    assert not saved_path.name.endswith("-.model")


def test_load_model_round_trip_preserves_hyperparameters(params, tokenizer, temp_data_dir):
    # GIVEN a saved model
    saved_path = save_model(params, tokenizer)

    # WHEN loading it back
    loaded_params, loaded_tokenizer = load_model(saved_path)

    # THEN core hyperparameters are restored
    assert loaded_params.embedding_dim == params.embedding_dim
    assert loaded_params.max_sequence_length == params.max_sequence_length
    assert loaded_params.transformer_block_count == params.transformer_block_count
    assert loaded_params.attn_head_count == params.attn_head_count
    assert loaded_tokenizer.vocab_size == tokenizer.vocab_size


def test_load_model_round_trip_preserves_parameter_values(params, tokenizer, temp_data_dir):
    # GIVEN a saved model
    saved_path = save_model(params, tokenizer)

    # WHEN loading it back
    loaded_params, _ = load_model(saved_path)

    # THEN every learnable parameter's value matches the original
    original = [p.value for p in params]
    restored = [p.value for p in loaded_params]
    assert original == restored


def test_load_model_round_trip_preserves_tokenizer_behavior(params, tokenizer, temp_data_dir):
    # GIVEN a saved model
    saved_path = save_model(params, tokenizer)

    # WHEN loading it back
    _, loaded_tokenizer = load_model(saved_path)

    # THEN the loaded tokenizer encodes and decodes identically to the original
    assert loaded_tokenizer.encode("abc") == tokenizer.encode("abc")
    assert loaded_tokenizer.decode([0, 1, 2]) == tokenizer.decode([0, 1, 2])
