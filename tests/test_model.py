import random

import pytest

from tests.helpers import maybe_import

AutoGradNode = maybe_import("zerogpt.autograd", "AutoGradNode")
GPTParams = maybe_import("zerogpt.model", "GPTParams")
create_random_matrix = maybe_import("zerogpt.model", "create_random_matrix")
gpt = maybe_import("zerogpt.model", "gpt")


@pytest.fixture
def small_params():
    return GPTParams.create(
        vocab_size=10,
        embedding_dim=8,
        max_sequence_length=4,
        transformer_block_count=2,
        attn_head_count=2,
        transformer_mlp_fanout_factor=4,
    )


def test_create_token_embedding_shape(small_params):
    # GIVEN params created with vocab_size=10, embedding_dim=8
    # WHEN inspecting the token embedding matrix
    rows = len(small_params.w_token_emb)
    cols = len(small_params.w_token_emb[0])

    # THEN it is `vocab_size` rows by `embedding_dim` columns
    assert rows == 10
    assert cols == 8


def test_create_position_embedding_shape(small_params):
    # GIVEN params created with max_sequence_length=4, embedding_dim=8
    # WHEN inspecting the position embedding matrix
    rows = len(small_params.w_position_emb)
    cols = len(small_params.w_position_emb[0])

    # THEN it is `max_sequence_length` rows by `embedding_dim` columns
    assert rows == 4
    assert cols == 8


def test_create_attn_matrices_count_matches_block_count(small_params):
    # GIVEN params created with transformer_block_count=2
    # WHEN inspecting the per-block attention matrix lists
    # THEN each list has one matrix per transformer block
    assert len(small_params.w_transformer_attn_q) == 2
    assert len(small_params.w_transformer_attn_k) == 2
    assert len(small_params.w_transformer_attn_v) == 2
    assert len(small_params.w_transformer_attn_out) == 2


def test_create_attn_matrices_are_square_embedding_dim(small_params):
    # GIVEN params created with embedding_dim=8
    # WHEN inspecting each Q matrix
    # THEN it is square with both dimensions equal to `embedding_dim`
    for block in small_params.w_transformer_attn_q:
        assert len(block) == 8
        assert len(block[0]) == 8


def test_create_mlp_fc1_has_fanout_rows(small_params):
    # GIVEN params created with embedding_dim=8, fanout=4
    # WHEN inspecting the MLP `fc1` matrix of each block
    # THEN it projects from `embedding_dim` up to `fanout * embedding_dim`
    for block in small_params.w_transformer_mlp_fc1:
        assert len(block) == 4 * 8
        assert len(block[0]) == 8


def test_create_mlp_fc2_has_fanout_cols(small_params):
    # GIVEN params created with embedding_dim=8, fanout=4
    # WHEN inspecting the MLP `fc2` matrix of each block
    # THEN it projects from `fanout * embedding_dim` back down to `embedding_dim`
    for block in small_params.w_transformer_mlp_fc2:
        assert len(block) == 8
        assert len(block[0]) == 4 * 8


def test_create_lm_head_shape(small_params):
    # GIVEN params created with vocab_size=10, embedding_dim=8
    # WHEN inspecting the LM head matrix
    rows = len(small_params.w_lm_head)
    cols = len(small_params.w_lm_head[0])

    # THEN it is `vocab_size` rows by `embedding_dim` columns
    assert rows == 10
    assert cols == 8


def test_vocab_size_property(small_params):
    # GIVEN params created with vocab_size=10
    # WHEN reading the `vocab_size` property
    vocab_size = small_params.vocab_size

    # THEN it reflects the construction argument
    assert vocab_size == 10


def test_embedding_dim_property(small_params):
    # GIVEN params created with embedding_dim=8
    # WHEN reading the `embedding_dim` property
    embedding_dim = small_params.embedding_dim

    # THEN it reflects the construction argument
    assert embedding_dim == 8


def test_max_sequence_length_property(small_params):
    # GIVEN params created with max_sequence_length=4
    # WHEN reading the `max_sequence_length` property
    max_sequence_length = small_params.max_sequence_length

    # THEN it reflects the construction argument
    assert max_sequence_length == 4


def test_attn_head_dim_is_embedding_dim_over_head_count(small_params):
    # GIVEN params with embedding_dim=8 and attn_head_count=2
    # WHEN reading `attn_head_dim`
    head_dim = small_params.attn_head_dim

    # THEN it equals `embedding_dim / attn_head_count`
    assert head_dim == 4


def test_transformer_block_count_property(small_params):
    # GIVEN params with transformer_block_count=2
    # WHEN reading the `transformer_block_count` property
    block_count = small_params.transformer_block_count

    # THEN it reflects the construction argument
    assert block_count == 2


def test_iter_yields_only_autograd_nodes(small_params):
    # GIVEN a `GPTParams` instance
    # WHEN iterating over it
    params = list(small_params)

    # THEN every yielded element is an `AutoGradNode`, ready for the optimizer
    assert all(isinstance(p, AutoGradNode) for p in params)


def test_iter_yields_expected_parameter_count(small_params):
    # GIVEN known hyperparameters used to build `small_params`
    vocab_size = 10
    embed_dim = 8
    seq_len = 4
    block_count = 2
    fanout = 4

    # WHEN iterating over the params
    yielded = list(small_params)

    # THEN the total count equals the sum of all matrix sizes
    expected_count = (
        # `w_token_emb`
        vocab_size * embed_dim
        # `w_position_emb`
        + seq_len * embed_dim
        # 4 attn matrices per block
        + block_count * 4 * embed_dim * embed_dim
        # `fc1` per block
        + block_count * fanout * embed_dim * embed_dim
        # `fc2` per block
        + block_count * embed_dim * fanout * embed_dim
        # `w_lm_head`
        + vocab_size * embed_dim
    )
    assert len(yielded) == expected_count


def test_create_with_zero_blocks_yields_empty_per_block_matrices():
    # GIVEN a request for zero transformer blocks
    # WHEN constructing the params
    params = GPTParams.create(
        vocab_size=5,
        embedding_dim=4,
        max_sequence_length=2,
        transformer_block_count=0,
        attn_head_count=1,
    )

    # THEN per-block matrix lists are empty
    assert params.w_transformer_attn_q == []
    assert params.w_transformer_mlp_fc1 == []


def _matrix(rows):
    return [[AutoGradNode(v) for v in row] for row in rows]


def _zero_out_block_weights(params):
    # Wipe every block weight so each block reduces to a residual skip.
    block_weight_matrices = (
        params.w_transformer_attn_q
        + params.w_transformer_attn_k
        + params.w_transformer_attn_v
        + params.w_transformer_attn_out
        + params.w_transformer_mlp_fc1
        + params.w_transformer_mlp_fc2
    )
    for matrix in block_weight_matrices:
        for row in matrix:
            for elem in row:
                elem.value = 0.0


def test_gpt_output_length_equals_vocab_size(small_params):
    # GIVEN an empty cache
    kv_cache = [[] for _ in range(small_params.transformer_block_count)]

    # WHEN running the forward pass
    logits = gpt(token_id=0, position_id=0, params=small_params, kv_cache=kv_cache)

    # THEN we get one logit per vocab entry
    assert len(logits) == 10


def test_gpt_appends_to_kv_cache_each_call(small_params):
    # GIVEN an empty cache
    kv_cache = [[] for _ in range(small_params.transformer_block_count)]

    # WHEN running two forward passes
    gpt(token_id=0, position_id=0, params=small_params, kv_cache=kv_cache)
    gpt(token_id=1, position_id=1, params=small_params, kv_cache=kv_cache)

    # THEN each block's cache has two entries
    for block_cache in kv_cache:
        assert len(block_cache) == 2


def test_gpt_output_is_zero_when_lm_head_is_zero(small_params):
    # GIVEN a zeroed LM head
    rows = len(small_params.w_lm_head)
    cols = len(small_params.w_lm_head[0])
    small_params.w_lm_head = [[AutoGradNode(0.0) for _ in range(cols)] for _ in range(rows)]
    kv_cache = [[] for _ in range(small_params.transformer_block_count)]

    # WHEN running the forward pass
    logits = gpt(token_id=0, position_id=0, params=small_params, kv_cache=kv_cache)

    # THEN every logit is zero
    for logit in logits:
        assert logit.value == 0.0


def test_gpt_with_zeroed_blocks_collapses_to_lm_head_over_rmsnorm():
    # GIVEN a tiny model with all block weights zeroed
    # (only embeddings + final norm + LM head remain)
    params = GPTParams.create(
        vocab_size=2,
        embedding_dim=2,
        max_sequence_length=1,
        transformer_block_count=1,
        attn_head_count=1,
        transformer_mlp_fanout_factor=1,
    )
    params.w_token_emb = _matrix([[3.0, 4.0], [9.0, 12.0]])
    params.w_position_emb = _matrix([[0.0, 0.0]])
    params.w_lm_head = _matrix([[1.0, 0.0], [0.0, 1.0]])
    _zero_out_block_weights(params)
    kv_cache = [[] for _ in range(params.transformer_block_count)]

    # WHEN running token 0 at position 0
    logits = gpt(token_id=0, position_id=0, params=params, kv_cache=kv_cache)

    # THEN with the identity LM head, the output is just rms_norm([3, 4])
    scaler = (12.5 + 1e-5) ** -0.5
    expected = [3.0 * scaler, 4.0 * scaler]
    for logit, e in zip(logits, expected, strict=True):
        assert logit.value == pytest.approx(e)


def test_gpt_position_embedding_changes_output():
    # GIVEN a zeroed-block model with two different position embeddings
    params = GPTParams.create(
        vocab_size=1,
        embedding_dim=2,
        max_sequence_length=2,
        transformer_block_count=1,
        attn_head_count=1,
        transformer_mlp_fanout_factor=1,
    )
    params.w_token_emb = _matrix([[1.0, 0.0]])
    params.w_position_emb = _matrix([[1.0, 0.0], [0.0, 1.0]])
    params.w_lm_head = _matrix([[1.0, 0.0]])
    _zero_out_block_weights(params)

    # WHEN running the same token at both positions (fresh cache each time)
    cache_0 = [[]]
    logits_at_pos_0 = gpt(token_id=0, position_id=0, params=params, kv_cache=cache_0)
    cache_1 = [[]]
    logits_at_pos_1 = gpt(token_id=0, position_id=1, params=params, kv_cache=cache_1)

    # THEN each output is the first component of rms_norm(token_emb + position_emb).
    # At pos 0, x = [2, 0]; at pos 1, x = [1, 1].
    expected_at_pos_0 = 2.0 * (2.0 + 1e-5) ** -0.5
    expected_at_pos_1 = 1.0 * (1.0 + 1e-5) ** -0.5
    assert logits_at_pos_0[0].value == pytest.approx(expected_at_pos_0)
    assert logits_at_pos_1[0].value == pytest.approx(expected_at_pos_1)


def test_gpt_single_entry_attention_writes_v_into_residual():
    # GIVEN a model where attention is the only active piece
    # W_Q = W_K = 0, so the softmax over the single cached entry is [1.0] and the head output is v.
    # W_V picks out the y-component of rms(x), W_out is identity, and the MLP weights are zero.
    params = GPTParams.create(
        vocab_size=2,
        embedding_dim=2,
        max_sequence_length=1,
        transformer_block_count=1,
        attn_head_count=1,
        transformer_mlp_fanout_factor=1,
    )
    params.w_token_emb = _matrix([[3.0, 4.0], [0.0, 0.0]])
    params.w_position_emb = _matrix([[0.0, 0.0]])
    params.w_lm_head = _matrix([[1.0, 0.0], [0.0, 1.0]])
    _zero_out_block_weights(params)
    params.w_transformer_attn_v[0] = _matrix([[0.0, 0.0], [0.0, 1.0]])
    params.w_transformer_attn_out[0] = _matrix([[1.0, 0.0], [0.0, 1.0]])
    kv_cache = [[]]

    # WHEN running the forward pass
    logits = gpt(token_id=0, position_id=0, params=params, kv_cache=kv_cache)

    # THEN, walking through the math:
    #     rms_in = [3, 4] / sqrt(12.5 + eps)
    #     v = [0, rms_in[1]]
    #     x_after_block = [3, 4 + rms_in[1]]
    #     output = rms_norm(x_after_block)
    eps = 1e-5
    initial_scaler = (12.5 + eps) ** -0.5
    rms_in_y = 4.0 * initial_scaler
    x_after_block = [3.0, 4.0 + rms_in_y]
    mean_squares = (x_after_block[0] ** 2 + x_after_block[1] ** 2) / 2
    final_scaler = (mean_squares + eps) ** -0.5
    expected = [x_after_block[0] * final_scaler, x_after_block[1] * final_scaler]
    for logit, e in zip(logits, expected, strict=True):
        assert logit.value == pytest.approx(e)


def test_create_random_matrix_shape():
    # GIVEN a request for a 4x3 matrix
    requested_rows, requested_cols = 4, 3

    # WHEN creating the matrix
    mat = create_random_matrix(rows=requested_rows, cols=requested_cols)

    # THEN the result has the requested number of rows and columns
    assert len(mat) == requested_rows
    assert all(len(row) == requested_cols for row in mat)


def test_create_random_matrix_returns_autograd_nodes():
    # GIVEN a request for a small matrix
    # WHEN creating it
    mat = create_random_matrix(rows=2, cols=2)

    # THEN every entry is an `AutoGradNode`, ready for autograd
    for row in mat:
        for elem in row:
            assert isinstance(elem, AutoGradNode)


def test_create_random_matrix_uses_provided_stddev(monkeypatch):
    # GIVEN a deterministic `random.gauss` that echoes its sigma argument
    monkeypatch.setattr(random, "gauss", lambda mu, sigma: sigma)

    # WHEN creating a matrix with stddev=0.05
    mat = create_random_matrix(rows=2, cols=2, stddev=0.05)

    # THEN every entry equals the stddev that was passed through to gauss
    for row in mat:
        for elem in row:
            assert elem.value == 0.05
