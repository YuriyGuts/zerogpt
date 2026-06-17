import os
import pickle
from pathlib import Path

from zerogpt.model import GPTParams
from zerogpt.tokenizer import Tokenizer

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
        ".model"
    )

    artifact = (gpt_params, tokenizer)
    artifact_save_path = data_dir / model_filename
    with open(artifact_save_path, "wb") as f:
        # TODO: don't do this in production.
        pickle.dump(artifact, f)

    return artifact_save_path


def load_model(path: Path) -> tuple[GPTParams, Tokenizer]:
    with open(path, "rb") as f:
        # TODO: don't do this in production.
        artifact = pickle.load(f)
    return artifact
