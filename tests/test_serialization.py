import json

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
    assert name.endswith(".json")


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

    # THEN no trailing `-` separator appears before the `.json` extension
    assert not saved_path.name.endswith("-.json")


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


def test_save_model_writes_valid_structured_json(params, tokenizer, temp_data_dir):
    # GIVEN a saved model
    saved_path = save_model(params, tokenizer)

    # WHEN reading the file back as plain JSON
    state = json.loads(saved_path.read_text(encoding="utf-8"))

    # THEN it is a self-describing, structured checkpoint
    assert state["format_version"] == 1
    assert state["attn_head_count"] == params.attn_head_count
    assert set(state["weights"]) == {
        "w_token_emb",
        "w_position_emb",
        "w_transformer_attn_q",
        "w_transformer_attn_k",
        "w_transformer_attn_v",
        "w_transformer_attn_out",
        "w_transformer_mlp_fc1",
        "w_transformer_mlp_fc2",
        "w_lm_head",
    }
    assert state["vocab"] == tokenizer.vocab


def test_load_model_rejects_unsupported_format_version(temp_data_dir):
    # GIVEN a checkpoint file declaring a future format version
    path = temp_data_dir / "future.json"
    path.write_text(json.dumps({"format_version": 999}), encoding="utf-8")

    # WHEN loading it THEN a clear error is raised
    with pytest.raises(ValueError, match="format version"):
        load_model(path)


def test_load_model_rejects_non_json_file(temp_data_dir):
    # GIVEN a file that is not valid JSON
    path = temp_data_dir / "corrupt.json"
    path.write_bytes(b"\x80\x04\x95not-json")

    # WHEN loading it THEN a friendly error is raised instead of crashing
    with pytest.raises(ValueError, match="not a valid"):
        load_model(path)


def test_load_model_round_trip_preserves_non_ascii_vocab(params, temp_data_dir):
    # GIVEN a tokenizer trained on Cyrillic text
    cyrillic_tokenizer = Tokenizer()
    cyrillic_tokenizer.train(["Київ", "Львів"])

    # WHEN saving and loading the model
    saved_path = save_model(params, cyrillic_tokenizer)
    _, loaded_tokenizer = load_model(saved_path)

    # THEN the Cyrillic vocabulary survives the round trip
    assert loaded_tokenizer.vocab == cyrillic_tokenizer.vocab
    assert loaded_tokenizer.encode("Київ") == cyrillic_tokenizer.encode("Київ")
