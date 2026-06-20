"""Training and inference loops."""

import itertools
import random
import time
from collections import deque

from zerogpt.autograd import AutoGradNode
from zerogpt.model import GPTParams
from zerogpt.model import gpt
from zerogpt.ops import log_softmax
from zerogpt.ops import softmax
from zerogpt.optimize import AdamOptimizer
from zerogpt.serialization import save_model
from zerogpt.tokenizer import Tokenizer

ATTN_HEAD_COUNT = 4
MLP_FANOUT_FACTOR = 4


def train(
    docs: list[str],
    iter_count: int,
    batch_size: int,
    learning_rate: float = 0.01,
    embedding_dim: int = 16,
    max_sequence_length: int = 20,
    transformer_block_count: int = 1,
    checkpoint_freq: int | None = None,
) -> tuple[GPTParams, Tokenizer]:
    """
    Train a model and tokenizer on a list of documents.

    Parameters
    ----------
    docs
        The training documents.
    iter_count
        The number of training iterations (batches).
    batch_size
        The number of documents per batch.
    learning_rate
        The learning rate of the optimizer.
    embedding_dim
        The embedding width (must be divisible by the attention head count).
    max_sequence_length
        The maximum sequence length (longer documents are truncated).
    transformer_block_count
        The number of transformer blocks.
    checkpoint_freq
        Save a checkpoint every N iterations, or None to disable.

    Returns
    -------
    The trained model parameters and tokenizer.

    Raises
    ------
    ValueError
        If `embedding_dim` is not divisible by the attention head count.
    """
    if embedding_dim % ATTN_HEAD_COUNT != 0:
        msg = (
            f"embedding_dim ({embedding_dim}) must be divisible by "
            f"attn_head_count ({ATTN_HEAD_COUNT})."
        )
        raise ValueError(msg)

    print("Training the tokenizer...")
    tokenizer = Tokenizer()
    tokenizer.train(docs)

    gpt_params = GPTParams.create(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=embedding_dim,
        max_sequence_length=max_sequence_length,
        transformer_block_count=transformer_block_count,
        attn_head_count=ATTN_HEAD_COUNT,
        transformer_mlp_fanout_factor=MLP_FANOUT_FACTOR,
    )
    optimizer = AdamOptimizer(
        params=list(gpt_params),
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.999,
    )

    print("===== Initial (untrained) predictions =====")
    make_sample_predictions(gpt_params, tokenizer, 10)

    print("Training the LLM...")
    tokenized_docs = [tokenizer.encode(doc)[: gpt_params.max_sequence_length] for doc in docs]
    random.shuffle(tokenized_docs)
    batch_loss_history = deque(maxlen=50)
    batches = list(itertools.batched(tokenized_docs, batch_size))

    for iter_idx in range(iter_count):
        iter_start_time = time.monotonic()
        batch_docs = batches[iter_idx % len(batches)]
        batch_losses = []

        for doc in batch_docs:
            kv_cache = gpt_params.create_kv_cache()

            # At each position, predict the next token and compute its loss.
            for position_id, token_id in enumerate(doc[:-1]):
                output_logits = gpt(
                    token_id=token_id,
                    position_id=position_id,
                    params=gpt_params,
                    kv_cache=kv_cache,
                )
                output_logprobs = log_softmax(output_logits)
                ground_truth_token_id = doc[position_id + 1]
                token_loss = -output_logprobs[ground_truth_token_id]
                batch_losses.append(token_loss)

        batch_loss = AutoGradNode.sum(batch_losses) / len(batch_losses)
        batch_loss.backpropagate()
        optimizer.step()

        batch_loss_history.append(batch_loss.value)
        moving_avg_batch_loss = sum(batch_loss_history) / len(batch_loss_history)

        iters_done = iter_idx + 1
        iter_duration = time.monotonic() - iter_start_time

        print(
            f"Iteration {iters_done:5d} / {iter_count:5d}"
            f" | Loss {batch_loss:8.5f}"
            f" | Avg({batch_loss_history.maxlen}) {moving_avg_batch_loss:8.5f}"
            f" | Duration {iter_duration:#.3} sec"
        )

        if checkpoint_freq and iters_done % checkpoint_freq == 0:
            print(f"===== Checkpoint after {iters_done} iterations =====")
            output_path = save_model(gpt_params, tokenizer, extra_id=f"iter-{iters_done:05d}")
            print(f"Model saved to {output_path}")
            make_sample_predictions(gpt_params, tokenizer, 10)

    return gpt_params, tokenizer


def predict(
    gpt_params: GPTParams,
    tokenizer: Tokenizer,
    prompt: str = "",
    temperature: float = 1.0,
) -> str:
    """
    Generate a continuation for the prompt by sampling one token at a time.

    Parameters
    ----------
    temperature
        The sampling temperature. Lower values make the output more deterministic.
        0 picks the single most likely token at every step (greedy decoding).

    Returns
    -------
    The initial prompt completed with the generated text.

    Raises
    ------
    ValueError
        If the prompt is longer than the model's maximum sequence length.
    """
    tokens = tokenizer.encode(prompt)[:-1]
    if len(tokens) > gpt_params.max_sequence_length:
        raise ValueError("Input sequence too long")

    kv_cache = gpt_params.create_kv_cache()

    position_id = 0
    while True:
        # Prefill.
        output_logits = gpt(
            token_id=tokens[position_id],
            position_id=position_id,
            params=gpt_params,
            kv_cache=kv_cache,
        )
        position_id += 1
        if position_id < len(tokens):
            continue

        # Decode.
        if temperature == 0:
            # Greedy sampling: deterministically pick the most likely token.
            next_token_id = max(
                range(gpt_params.vocab_size),
                key=lambda token_id: output_logits[token_id].value,
            )
        else:
            # Probabilistic sampling: sample the next token using the predictions as weights.
            output_probs = softmax([elem / temperature for elem in output_logits])
            output_probs = [elem.value for elem in output_probs]
            next_token_id = random.choices(
                population=range(gpt_params.vocab_size),
                weights=output_probs,
                k=1,
            )[0]
        tokens.append(next_token_id)
        if next_token_id == tokenizer.eos_token or len(tokens) >= gpt_params.max_sequence_length:
            break

    return tokenizer.decode(tokens)


def make_sample_predictions(
    gpt_params: GPTParams,
    tokenizer: Tokenizer,
    sample_size: int,
) -> None:
    """Print some sampled generations at different temperatures."""
    for temperature in (1.0, 0.5):
        print(f"----- t={temperature:.1f} -----")
        for _ in range(sample_size):
            prediction = predict(
                gpt_params=gpt_params,
                tokenizer=tokenizer,
                prompt="",
                temperature=temperature,
            )
            print(prediction)
