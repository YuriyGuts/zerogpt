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


def train(
    docs: list[str],
    iter_count: int,
    batch_size: int,
    checkpoint_freq: int | None = None,
) -> tuple[GPTParams, Tokenizer]:
    print("Training the tokenizer...")
    tokenizer = Tokenizer()
    tokenizer.train(docs)

    gpt_params = GPTParams.create(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=16,
        max_sequence_length=20,
        transformer_block_count=1,
        attn_head_count=4,
        transformer_mlp_fanout_factor=4,
    )
    optimizer = AdamOptimizer(
        params=list(gpt_params),
        learning_rate=0.01,
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
        batch_loss_history.append(batch_loss.value)
        moving_avg_batch_loss = sum(batch_loss_history) / len(batch_loss_history)
        optimizer.step()

        iters_done = iter_idx + 1
        iter_duration = time.monotonic() - iter_start_time

        print(
            f"Iter {iters_done} / {iter_count}"
            f" | Loss {batch_loss:10.5f}"
            f" | Avg({batch_loss_history.maxlen}) {moving_avg_batch_loss:10.5f}"
            f" | Duration {iter_duration:.3} sec"
        )

        if checkpoint_freq and iters_done % checkpoint_freq == 0:
            print(f"===== Checkpoint after {iters_done} iterations =====")
            output_path = save_model(gpt_params, tokenizer, extra_id=f"iter-{iters_done}")
            print(f"Model saved to {output_path}")
            make_sample_predictions(gpt_params, tokenizer, 10)

    return gpt_params, tokenizer


def predict(
    gpt_params: GPTParams,
    tokenizer: Tokenizer,
    prompt: str = "",
    temperature: float = 1.0,
) -> str:
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
