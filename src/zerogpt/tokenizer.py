"""A character-level tokenizer."""

from __future__ import annotations


class Tokenizer:
    """A character-level tokenizer with BOS, EOS, and UNK special tokens."""

    def __init__(self) -> None:
        self._vocab = []
        self._char_to_token_id = {}

    @classmethod
    def from_vocab(cls, vocab: list[str]) -> Tokenizer:
        """Build a tokenizer from a previously trained vocabulary."""
        tokenizer = cls()
        tokenizer._vocab = list(vocab)
        tokenizer._char_to_token_id = {
            char: token_id for token_id, char in enumerate(tokenizer._vocab)
        }
        return tokenizer

    def _ensure_trained(self) -> None:
        if not self._vocab:
            raise RuntimeError("Tokenizer must be trained first")

    @property
    def vocab(self) -> list[str]:
        """The trained vocabulary, as a list of characters."""
        self._ensure_trained()
        return list(self._vocab)

    @property
    def vocab_size(self) -> int:
        """The vocabulary size, including the special tokens."""
        self._ensure_trained()
        return len(self._vocab) + len(self.special_tokens)

    @property
    def bos_token(self) -> int:
        """The beginning-of-sequence token id."""
        self._ensure_trained()
        return len(self._vocab)

    @property
    def eos_token(self) -> int:
        """The end-of-sequence token id."""
        self._ensure_trained()
        return len(self._vocab) + 1

    @property
    def unk_token(self) -> int:
        """The unknown-character token id."""
        self._ensure_trained()
        return len(self._vocab) + 2

    @property
    def special_tokens(self) -> tuple[int, ...]:
        """The ids of all special tokens."""
        return (self.bos_token, self.eos_token, self.unk_token)

    def train(self, docs: list[str]) -> None:
        """Build the vocabulary from the characters seen in the documents."""
        self._vocab = sorted({char for doc in docs for char in doc})
        self._char_to_token_id = {char: token_id for token_id, char in enumerate(self._vocab)}

    def encode(self, doc: str) -> list[int]:
        """Encode a string into token ids, wrapped in BOS and EOS."""
        self._ensure_trained()
        return (
            [self.bos_token]
            + [self._char_to_token_id.get(char, self.unk_token) for char in doc]
            + [self.eos_token]
        )

    def decode(self, token_ids: list[int]) -> str:
        """Decode token ids back into a string, dropping special tokens."""
        self._ensure_trained()
        return "".join(
            self._vocab[token_id] for token_id in token_ids if token_id not in self.special_tokens
        )
