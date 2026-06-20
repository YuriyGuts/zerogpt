"""Command-line interface for training and running the model."""

import argparse
import gc
import random
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from zerogpt import runner
from zerogpt import serialization

DEFAULT_DATA_PATH = serialization.default_data_dir / "ua-settlement-names.txt"


@contextmanager
def gc_disabled() -> Iterator[None]:
    """
    Temporarily disable the garbage collector to speed up training.

    Experiments show that creating a lot of `AutoGradNode` objects often triggers the GC,
    which fails to reclaim any objects and effectively just wastes time.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with the `train` and `predict` subcommands."""
    parser = argparse.ArgumentParser(prog="zerogpt", description="Train and run a tiny GPT.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model on a text corpus.")
    train_parser.add_argument(
        "data",
        help="The training corpus file, one document per line (default: the bundled dataset).",
        nargs="?",
        type=Path,
        default=DEFAULT_DATA_PATH,
    )
    train_parser.add_argument(
        "--iterations",
        help="The number of training iterations.",
        type=int,
        default=1000,
    )
    train_parser.add_argument(
        "--batch-size",
        help="The number of documents to process in each training iteration.",
        type=int,
        default=32,
    )
    train_parser.add_argument(
        "--learning-rate",
        help="The learning rate of the optimizer.",
        type=float,
        default=0.01,
    )
    train_parser.add_argument(
        "--embedding-dim",
        help=f"Embedding dimension (must be divisible by {runner.ATTN_HEAD_COUNT}).",
        type=int,
        default=16,
    )
    train_parser.add_argument(
        "--context-length",
        help="The maximum sequence length the model can process (longer documents get truncated).",
        type=int,
        default=20,
    )
    train_parser.add_argument(
        "--blocks",
        help="The number of transformer blocks.",
        type=int,
        default=1,
    )
    train_parser.add_argument(
        "--seed",
        help="The random seed for reproducible training.",
        type=int,
        default=1234,
    )
    train_parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=20,
        help="Save a checkpoint every N iterations (0 disables checkpointing).",
    )
    train_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: an auto-named file in the data directory).",
    )

    predict_parser = subparsers.add_parser("predict", help="Generate text from a saved model.")
    predict_parser.add_argument(
        "model",
        help="The path to a saved model checkpoint.",
        type=Path,
    )
    predict_parser.add_argument(
        "--prompt",
        help="Generate once for this prompt and exit instead of starting an interactive session.",
        type=str,
        default=None,
    )
    predict_parser.add_argument(
        "--temperature",
        help="The temperature for next token sampling.",
        type=float,
        default=0.5,
    )
    predict_parser.add_argument(
        "--seed",
        help="The random seed for reproducible sampling.",
        type=int,
        default=None,
    )

    return parser


def run_train(args: argparse.Namespace) -> None:
    """Run the `train` subcommand."""
    random.seed(args.seed)

    try:
        docs = args.data.read_text(encoding="utf-8").splitlines(keepends=False)
    except OSError as exc:
        print(f"Cannot read dataset: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    with gc_disabled():
        gpt_params, tokenizer = runner.train(
            docs=docs,
            iter_count=args.iterations,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            embedding_dim=args.embedding_dim,
            max_sequence_length=args.context_length,
            transformer_block_count=args.blocks,
            checkpoint_freq=args.checkpoint_freq or None,
        )

    saved_path = serialization.save_model(gpt_params, tokenizer, path=args.output)
    print(f"Final model saved to {saved_path}")


def run_predict(args: argparse.Namespace) -> None:
    """Run the `predict` subcommand, interactively or for a single prompt."""
    if args.seed is not None:
        random.seed(args.seed)

    try:
        gpt_params, tokenizer = serialization.load_model(args.model)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Cannot load model: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.prompt is not None:
        try:
            prediction = runner.predict(
                gpt_params=gpt_params,
                tokenizer=tokenizer,
                prompt=args.prompt,
                temperature=args.temperature,
            )
        except Exception as exc:
            print(exc, file=sys.stderr)
            raise SystemExit(1) from exc
        print(prediction)
        return

    while True:
        try:
            prompt = input("Enter your prompt: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        try:
            prediction = runner.predict(
                gpt_params=gpt_params,
                tokenizer=tokenizer,
                prompt=prompt,
                temperature=args.temperature,
            )
        except Exception as exc:
            print(exc, file=sys.stderr)
            continue
        print(prediction)


def main(argv: list[str] | None = None) -> None:
    """Parse command-line arguments and run the chosen subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        run_train(args)
    elif args.command == "predict":
        run_predict(args)


if __name__ == "__main__":
    main()
