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
) -> Path:
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

    artifact = {
        "format_version": FORMAT_VERSION,
        "vocab": tokenizer.vocab,
        **gpt_params.to_dict(),
    }
    artifact_save_path = data_dir / model_filename
    with open(artifact_save_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, allow_nan=False)

    return artifact_save_path


def load_model(path: Path) -> tuple[GPTParams, Tokenizer]:
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
