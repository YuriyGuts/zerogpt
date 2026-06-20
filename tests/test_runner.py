import pytest

from tests.helpers import maybe_import

train = maybe_import("zerogpt.runner", "train")


def test_train_rejects_indivisible_embedding_dim():
    # GIVEN an embedding dim not divisible by the attention head count
    # WHEN training
    # THEN it fails fast with a clear error (before any real work)
    with pytest.raises(ValueError, match="divisible"):
        train(docs=["abc"], iter_count=1, batch_size=1, embedding_dim=30)
