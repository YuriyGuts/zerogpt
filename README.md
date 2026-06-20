# zerogpt

An educational, zero-dependency training/inference pipeline for a GPT-like LLM. Inspired by [Andrej Karpathy's microgpt](https://karpathy.github.io/2026/02/12/microgpt/).

[![License](https://img.shields.io/badge/license-BSD--3--Clause-green)](LICENSE)

The goal of this project is to build a working GPT completely from scratch. It is a learning exercise in understanding what actually happens inside an LLM down to bare scalar operations.

Most of the project was [live-coded](https://www.youtube.com/watch?v=jAq4sLP033k&pp=0gcJCT8LAYcqIYzv), with some offline polishing afterward.

## How It Works

A character-level tokenizer + a small GPT-2-style transformer:
* Learnable token and positional embeddings.
* A stack of transformer blocks (RMSNorm + multi-head attention with a KV cache + MLP).
* A final language modeling head.

For training, we use a plain Adam optimizer on top of a handwritten autograd engine.

The bundled dataset is ~17k Ukrainian settlement names, so a trained model learns to invent plausible-sounding names.
You can also point it at any text file (one document per line) instead.

Since every operation runs on a single Python thread, on scalar floats, one token at a time, it is slow and the models are tiny (~5k parameters).
However, that's also the point: the code stays readable, and the model is meant to be understood rather than deployed.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```shell
uv sync
```

## Usage

### Training

Train on the bundled dataset with the default settings:

```shell
uv run zerogpt train
```

You can pass your own corpus and tune the model and training loop:

```shell
uv run zerogpt train my-corpus.txt \
    --iterations 2000 \
    --batch-size 32 \
    --learning-rate 0.01 \
    --embedding-dim 16 \
    --context-length 20 \
    --blocks 1 \
    --output my-model.json
```

Checkpoints are saved periodically (see `--checkpoint-freq`). Run `uv run zerogpt train -h` for the full list of options.

### Generating

Point `predict` at a saved checkpoint to start an interactive prompt.

A pretrained demo checkpoint for Ukrainian settlement names is available in the `data` directory:

```shell
uv run zerogpt predict data/gpt-demo-checkpoint.json
```

```
Enter your prompt: ТЕСТ
ТЕСТІВКА
Enter your prompt: _
```

Type a prompt and the model continues it. Press Ctrl-D to quit.

For a single non-interactive generation, use `--prompt`:

```shell
uv run zerogpt predict data/gpt-demo-checkpoint.json --prompt "ТЕСТ" --temperature 0.5
```

Checkpoints are plain JSON, so they are safe to share and easy to inspect.

## Development

```shell
make lint           # Check code formatting, linter issues, and types
make lint-fix       # Auto-fix code formatting and linter issues
make test           # Run tests
make test-coverage  # Run tests with coverage report
make check          # Lint + test
```

## License

The source code is licensed under the [BSD-3-Clause License](LICENSE).
