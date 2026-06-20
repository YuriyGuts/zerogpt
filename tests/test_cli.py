from pathlib import Path

import pytest

from zerogpt import cli
from zerogpt.cli import build_parser
from zerogpt.cli import main


def test_train_parser_defaults():
    # GIVEN only the train subcommand
    args = build_parser().parse_args(["train"])

    # THEN every default matches the documented value
    assert args.command == "train"
    assert args.data == cli.DEFAULT_DATA_PATH
    assert args.iterations == 1000
    assert args.batch_size == 32
    assert args.learning_rate == 0.01
    assert args.embedding_dim == 16
    assert args.context_length == 20
    assert args.blocks == 1
    assert args.seed == 1234
    assert args.checkpoint_freq == 20
    assert args.output is None


def test_train_parser_overrides():
    # GIVEN explicit overrides for every train flag
    args = build_parser().parse_args(
        [
            "train",
            "corpus.txt",
            "--iterations",
            "5",
            "--batch-size",
            "8",
            "--learning-rate",
            "0.05",
            "--embedding-dim",
            "32",
            "--context-length",
            "40",
            "--blocks",
            "2",
            "--seed",
            "7",
            "--checkpoint-freq",
            "0",
            "--output",
            "out.json",
        ]
    )

    # THEN the parsed values reflect the overrides
    assert args.data == Path("corpus.txt")
    assert args.iterations == 5
    assert args.batch_size == 8
    assert args.learning_rate == 0.05
    assert args.embedding_dim == 32
    assert args.context_length == 40
    assert args.blocks == 2
    assert args.seed == 7
    assert args.checkpoint_freq == 0
    assert args.output == Path("out.json")


def test_predict_parser_defaults():
    # GIVEN the predict subcommand with only the required model path
    args = build_parser().parse_args(["predict", "model.json"])

    # THEN the model path is parsed and the rest take their defaults
    assert args.command == "predict"
    assert args.model == Path("model.json")
    assert args.prompt is None
    assert args.temperature == 0.5
    assert args.seed is None


def test_predict_parser_accepts_prompt():
    # GIVEN a one-shot prompt
    args = build_parser().parse_args(["predict", "model.json", "--prompt", "Ky"])

    # THEN it is captured for non-interactive generation
    assert args.prompt == "Ky"


def test_parser_requires_subcommand():
    # GIVEN no subcommand THEN parsing exits
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_predict_parser_requires_model():
    # GIVEN predict without a model path THEN parsing exits
    with pytest.raises(SystemExit):
        build_parser().parse_args(["predict"])


def test_train_dispatch_maps_flags_to_runner(tmp_path, monkeypatch):
    # GIVEN a data file plus stubbed train/save so no real work happens
    data_file = tmp_path / "corpus.txt"
    data_file.write_text("abc\ndef\n", encoding="utf-8")

    recorded = {}

    def fake_train(**kwargs):
        recorded.update(kwargs)
        return object(), object()

    saved = {}

    def fake_save_model(gpt_params, tokenizer, extra_id="", path=None):
        saved["path"] = path
        return path or (tmp_path / "auto.json")

    monkeypatch.setattr("zerogpt.runner.train", fake_train)
    monkeypatch.setattr("zerogpt.serialization.save_model", fake_save_model)

    # WHEN running the train subcommand with explicit flags
    out_path = tmp_path / "out.json"
    main(
        [
            "train",
            str(data_file),
            "--iterations",
            "2",
            "--batch-size",
            "8",
            "--learning-rate",
            "0.05",
            "--embedding-dim",
            "32",
            "--context-length",
            "40",
            "--blocks",
            "2",
            "--checkpoint-freq",
            "0",
            "--output",
            str(out_path),
        ]
    )

    # THEN the CLI flags map onto the right runner.train kwargs
    assert recorded["docs"] == ["abc", "def"]
    assert recorded["iter_count"] == 2
    assert recorded["batch_size"] == 8
    assert recorded["learning_rate"] == 0.05
    assert recorded["embedding_dim"] == 32
    assert recorded["max_sequence_length"] == 40
    assert recorded["transformer_block_count"] == 2
    assert recorded["checkpoint_freq"] is None  # 0 disables checkpointing
    assert saved["path"] == out_path


def test_predict_one_shot_prints_result(monkeypatch, capsys):
    # GIVEN a stubbed model and generator
    monkeypatch.setattr("zerogpt.serialization.load_model", lambda path: (object(), object()))

    recorded = {}

    def fake_predict(gpt_params, tokenizer, prompt="", temperature=1.0):
        recorded["prompt"] = prompt
        recorded["temperature"] = temperature
        return "kyiv"

    monkeypatch.setattr("zerogpt.runner.predict", fake_predict)

    # WHEN running a one-shot prediction
    main(["predict", "model.json", "--prompt", "Ky", "--temperature", "0.3"])

    # THEN the result is printed and the args are forwarded
    assert "kyiv" in capsys.readouterr().out
    assert recorded["prompt"] == "Ky"
    assert recorded["temperature"] == 0.3


def test_predict_load_failure_exits_nonzero(monkeypatch, capsys):
    # GIVEN a checkpoint that fails to load
    def fake_load(path):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr("zerogpt.serialization.load_model", fake_load)

    # WHEN predicting
    # THEN it exits non-zero with a clear message
    with pytest.raises(SystemExit) as excinfo:
        main(["predict", "missing.json"])
    assert excinfo.value.code == 1
    assert "Cannot load model" in capsys.readouterr().err


def test_predict_repl_exits_cleanly_on_eof(monkeypatch):
    # GIVEN a loadable model and an immediate EOF (Ctrl-D) at the prompt
    monkeypatch.setattr("zerogpt.serialization.load_model", lambda path: (object(), object()))

    def fake_input(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)

    # WHEN entering the interactive loop
    # THEN it returns without raising
    main(["predict", "model.json"])
