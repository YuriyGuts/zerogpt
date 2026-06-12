from pathlib import Path

from zerogpt.autograd import AutoGradNode
from zerogpt.tokenizer import Tokenizer


def main() -> None:
    data_dir = Path(__file__).parent.parent.parent / "data"
    dataset_path = data_dir / "ua-settlement-names.txt"
    docs = dataset_path.read_text().splitlines(keepends=False)

    print("--- Tokenizer ---")
    tokenizer = Tokenizer()
    tokenizer.train(docs)
    lviv_encoded = tokenizer.encode("ЛЬВІВ")
    lviv_decoded = tokenizer.decode(lviv_encoded)

    print(f"encode(ЛЬВІВ): {lviv_encoded}")
    print(f"decode({lviv_encoded}): {lviv_decoded}")

    print()
    print("--- Autograd ---")
    a = AutoGradNode(2.0)
    b = AutoGradNode(3.0)
    f = a + b
    g = a * f
    h = a - g
    h.backpropagate()

    print(f"a) value: {a.value:7.2f}  grad: {a.grad:7.2f}")
    print(f"b) value: {b.value:7.2f}  grad: {b.grad:7.2f}")
    print(f"f) value: {f.value:7.2f}  grad: {f.grad:7.2f}")
    print(f"g) value: {g.value:7.2f}  grad: {g.grad:7.2f}")
    print(f"h) value: {h.value:7.2f}  grad: {h.grad:7.2f}")


if __name__ == "__main__":
    main()
