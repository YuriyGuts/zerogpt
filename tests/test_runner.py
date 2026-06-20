import math
import random

import pytest

from zerogpt.model import GPTParams
from zerogpt.model import gpt
from zerogpt.ops import log_softmax
from zerogpt.runner import make_sample_predictions
from zerogpt.runner import predict
from zerogpt.runner import train
from zerogpt.tokenizer import Tokenizer


def _untrained_model(tokenizer, max_sequence_length=10):
    return GPTParams.create(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=8,
        max_sequence_length=max_sequence_length,
        transformer_block_count=1,
    )


def _avg_next_token_nll(params, tokenizer, docs):
    # Replay each document through the model and average the next-token negative log-likelihood.
    total = 0.0
    count = 0
    for doc in docs:
        tokens = tokenizer.encode(doc)[: params.max_sequence_length]
        kv_cache = params.create_kv_cache()
        for position_id, token_id in enumerate(tokens[:-1]):
            logits = gpt(
                token_id=token_id, position_id=position_id, params=params, kv_cache=kv_cache
            )
            logprobs = log_softmax(logits)
            total += -logprobs[tokens[position_id + 1]].value
            count += 1
    return total / count


def test_train_rejects_indivisible_embedding_dim():
    # GIVEN an embedding dim not divisible by the attention head count
    # WHEN training
    # THEN it fails fast with a clear error (before any real work)
    with pytest.raises(ValueError, match="divisible"):
        train(docs=["abc"], iter_count=1, batch_size=1, embedding_dim=30)


def test_train_returns_model_sized_to_args():
    # GIVEN a tiny corpus
    docs = ["abc", "cab", "bca"]

    # WHEN training for a single iteration with an explicit architecture
    params, tokenizer = train(
        docs=docs,
        iter_count=1,
        batch_size=2,
        embedding_dim=8,
        max_sequence_length=6,
        transformer_block_count=2,
    )

    # THEN the model matches the requested architecture
    assert params.embedding_dim == 8
    assert params.max_sequence_length == 6
    assert params.transformer_block_count == 2
    assert params.attn_head_count == 4

    # AND the tokenizer is trained on the corpus characters
    assert tokenizer.vocab == ["a", "b", "c"]
    assert params.vocab_size == tokenizer.vocab_size


def test_train_reduces_loss_below_uniform():
    # GIVEN a tiny corpus the model can easily memorize
    random.seed(0)
    docs = ["abab", "baba"]

    # WHEN training for a while
    params, tokenizer = train(
        docs=docs,
        iter_count=80,
        batch_size=2,
        learning_rate=0.05,
        embedding_dim=8,
        max_sequence_length=8,
        transformer_block_count=1,
    )

    # THEN the model predicts the training data far better than uniform chance
    nll = _avg_next_token_nll(params, tokenizer, docs)
    assert nll < math.log(params.vocab_size)


def test_train_writes_checkpoints_when_frequency_set(tmp_path, monkeypatch):
    # GIVEN a redirected data directory
    monkeypatch.setattr("zerogpt.serialization.data_dir", tmp_path)

    # WHEN training with a checkpoint every iteration
    train(
        docs=["abc", "cab"],
        iter_count=2,
        batch_size=2,
        embedding_dim=8,
        max_sequence_length=6,
        transformer_block_count=1,
        checkpoint_freq=1,
    )

    # THEN one checkpoint file is written per iteration
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_train_writes_no_checkpoints_when_disabled(tmp_path, monkeypatch):
    # GIVEN a redirected data directory
    monkeypatch.setattr("zerogpt.serialization.data_dir", tmp_path)

    # WHEN training without checkpointing
    train(
        docs=["abc"],
        iter_count=2,
        batch_size=1,
        embedding_dim=8,
        max_sequence_length=6,
        transformer_block_count=1,
        checkpoint_freq=None,
    )

    # THEN nothing is written
    assert list(tmp_path.glob("*.json")) == []


def test_predict_continues_the_prompt():
    # GIVEN an (untrained) model and tokenizer
    random.seed(0)
    tokenizer = Tokenizer()
    tokenizer.train(["abab", "baba"])
    params = _untrained_model(tokenizer)

    # WHEN generating from a prompt
    result = predict(params, tokenizer, prompt="ab", temperature=1.0)

    # THEN the output starts with the prompt and stays within the context window
    assert result.startswith("ab")
    assert len(result) <= params.max_sequence_length


def test_predict_rejects_prompt_longer_than_context():
    # GIVEN a model with a very short context window
    tokenizer = Tokenizer()
    tokenizer.train(["abcde"])
    params = _untrained_model(tokenizer, max_sequence_length=3)

    # WHEN the prompt does not fit
    # THEN predicting raises a clear error
    with pytest.raises(ValueError, match="too long"):
        predict(params, tokenizer, prompt="abcde")


def test_make_sample_predictions_prints_each_sample(capsys):
    # GIVEN an (untrained) model and tokenizer
    random.seed(0)
    tokenizer = Tokenizer()
    tokenizer.train(["abc"])
    params = _untrained_model(tokenizer, max_sequence_length=6)

    # WHEN printing samples
    make_sample_predictions(params, tokenizer, sample_size=3)

    # THEN there is a header per temperature, each followed by the requested number of samples
    lines = capsys.readouterr().out.splitlines()
    headers = [line for line in lines if line.startswith("-----")]
    assert len(headers) == 2
    assert len(lines) == 2 + 2 * 3
