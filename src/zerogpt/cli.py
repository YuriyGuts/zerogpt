import gc
import random
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from zerogpt.runner import predict
from zerogpt.runner import train
from zerogpt.serialization import load_model


@contextmanager
def gc_disabled() -> Iterator[None]:
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def main() -> None:
    random.seed(1234)
    data_dir = Path(__file__).parent.parent.parent / "data"
    dataset_path = data_dir / "ua-settlement-names.txt"
    docs = dataset_path.read_text().splitlines(keepends=False)

    gpt_params, tokenizer = load_model(
        data_dir / "gpt-vocab-38-seq-20-emb-16-transf-1-attn-4.model"
    )

    with gc_disabled():
        train(
            docs=docs,
            iter_count=1000,
            batch_size=32,
            checkpoint_freq=20,
        )

    while True:
        prompt = input("Enter your prompt: ")
        prediction = predict(
            gpt_params=gpt_params,
            tokenizer=tokenizer,
            prompt=prompt,
            temperature=0.5,
        )
        print(prediction)


if __name__ == "__main__":
    main()
