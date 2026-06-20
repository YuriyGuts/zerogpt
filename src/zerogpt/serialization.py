"""Save and load models as plain JSON checkpoints."""

import json
import os
from pathlib import Path

from zerogpt.model import GPTParams
from zerogpt.tokenizer import Tokenizer

FORMAT_VERSION = 1

default_data_dir = Path(__file__).parent.parent.parent / "data"
data_dir = Path(os.environ.get("ZEROGPT_DATA_DIR") or default_data_dir)


def save_model(
    gpt_params: GPTParams,
    tokenizer: Tokenizer,
    extra_id: str = "",
    path: Path | None = None,
) -> Path:
    """
    Save a model and its tokenizer to a JSON checkpoint.

    Parameters
    ----------
    extra_id
        An optional suffix for the auto-generated filename (e.g. a checkpoint id).
    path
        An explicit output path. When omitted, the file is auto-named in the data directory.

    Returns
    -------
    The path the checkpoint was written to.
    """
    if path is None:
        model_filename = (
            "gpt"
            f"-vocab-{gpt_params.vocab_size}"
            f"-seq-{gpt_params.max_sequence_length}"
            f"-emb-{gpt_params.embedding_dim}"
            f"-transf-{gpt_params.transformer_block_count}"
            f"-attn-{gpt_params.attn_head_count}"
            f"{'-' + extra_id if extra_id else ''}"
            ".json"
        )
        path = data_dir / model_filename

    artifact = {
        "format_version": FORMAT_VERSION,
        "vocab": tokenizer.vocab,
        **gpt_params.to_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, allow_nan=False)

    return path


def load_model(path: Path) -> tuple[GPTParams, Tokenizer]:
    """
    Load a model and tokenizer from a JSON checkpoint.

    Returns
    -------
    The restored model parameters and tokenizer.

    Raises
    ------
    ValueError
        If the file is not a valid checkpoint or has an unsupported format version.
    """
    try:
        with open(path, encoding="utf-8") as f:
            artifact = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path} is not a valid checkpoint.") from exc

    if not isinstance(artifact, dict):
        raise ValueError(f"{path} is not a valid checkpoint.")

    format_version = artifact.get("format_version")
    if format_version != FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format version: {format_version}.")

    gpt_params = GPTParams.from_dict(artifact)
    tokenizer = Tokenizer.from_vocab(artifact["vocab"])
    return gpt_params, tokenizer
