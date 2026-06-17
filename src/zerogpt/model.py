import random
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import chain

from zerogpt.autograd import AutoGradNode
from zerogpt.autograd import Matrix
from zerogpt.autograd import Vector
from zerogpt.ops import linear
from zerogpt.ops import rms_norm
from zerogpt.ops import softmax
from zerogpt.ops import vec_dot_product
from zerogpt.ops import vec_sum


def create_random_matrix(rows: int, cols: int, stddev: float = 0.02) -> Matrix:
    return [
        [AutoGradNode(random.gauss(mu=0.0, sigma=stddev)) for _ in range(cols)] for _ in range(rows)
    ]


@dataclass(slots=True)
class GPTParams:
    @property
    def embedding_dim(self) -> int:
        return len(self.w_token_emb[0])

    @property
    def vocab_size(self) -> int:
        return len(self.w_token_emb)

    @property
    def max_sequence_length(self) -> int:
        return len(self.w_position_emb)

    @property
    def transformer_block_count(self) -> int:
        return len(self.w_transformer_attn_q)

    @property
    def attn_head_dim(self) -> int:
        return self.embedding_dim // self.attn_head_count

    attn_head_count: int

    w_token_emb: Matrix
    w_position_emb: Matrix

    w_transformer_attn_q: list[Matrix]
    w_transformer_attn_k: list[Matrix]
    w_transformer_attn_v: list[Matrix]
    w_transformer_attn_out: list[Matrix]
    w_transformer_mlp_fc1: list[Matrix]
    w_transformer_mlp_fc2: list[Matrix]

    w_lm_head: Matrix

    @classmethod
    def create(
        cls,
        vocab_size: int,
        embedding_dim: int,
        max_sequence_length: int,
        transformer_block_count: int,
        attn_head_count: int = 4,
        transformer_mlp_fanout_factor: int = 4,
    ):
        return GPTParams(
            attn_head_count=attn_head_count,
            w_token_emb=create_random_matrix(vocab_size, embedding_dim),
            w_position_emb=create_random_matrix(max_sequence_length, embedding_dim),
            w_transformer_attn_q=[
                create_random_matrix(embedding_dim, embedding_dim)
                for _ in range(transformer_block_count)
            ],
            w_transformer_attn_k=[
                create_random_matrix(embedding_dim, embedding_dim)
                for _ in range(transformer_block_count)
            ],
            w_transformer_attn_v=[
                create_random_matrix(embedding_dim, embedding_dim)
                for _ in range(transformer_block_count)
            ],
            w_transformer_attn_out=[
                create_random_matrix(embedding_dim, embedding_dim)
                for _ in range(transformer_block_count)
            ],
            w_transformer_mlp_fc1=[
                create_random_matrix(embedding_dim * transformer_mlp_fanout_factor, embedding_dim)
                for _ in range(transformer_block_count)
            ],
            w_transformer_mlp_fc2=[
                create_random_matrix(embedding_dim, embedding_dim * transformer_mlp_fanout_factor)
                for _ in range(transformer_block_count)
            ],
            w_lm_head=create_random_matrix(vocab_size, embedding_dim),
        )

    def __iter__(self) -> Iterator[AutoGradNode]:
        yield from chain(
            chain.from_iterable(self.w_token_emb),
            chain.from_iterable(self.w_position_emb),
            chain.from_iterable(chain.from_iterable(self.w_transformer_attn_q)),
            chain.from_iterable(chain.from_iterable(self.w_transformer_attn_k)),
            chain.from_iterable(chain.from_iterable(self.w_transformer_attn_v)),
            chain.from_iterable(chain.from_iterable(self.w_transformer_attn_out)),
            chain.from_iterable(chain.from_iterable(self.w_transformer_mlp_fc1)),
            chain.from_iterable(chain.from_iterable(self.w_transformer_mlp_fc2)),
            chain.from_iterable(self.w_lm_head),
        )


def gpt(
    token_id: int,
    position_id: int,
    params: GPTParams,
    # [block_idx][position_id] -> (k, v)
    kv_cache: list[list[tuple[Vector, Vector]]],
) -> Vector:
    token_emb = params.w_token_emb[token_id]
    position_emb = params.w_position_emb[position_id]
    x = vec_sum(token_emb, position_emb)

    # Transformer Blocks.
    for block_idx in range(params.transformer_block_count):
        x_residual = x
        x = rms_norm(x)

        # Multi-Head Attention.
        query = linear(x, params.w_transformer_attn_q[block_idx])
        key = linear(x, params.w_transformer_attn_k[block_idx])
        value = linear(x, params.w_transformer_attn_v[block_idx])
        kv_cache[block_idx].append((key, value))

        # Query  ***|***|***|***
        # Key    ***|***|***|***
        # Value  ***|***|***|***
        #        ^ head 0    ^ head 3
        attn_output = []

        # Attention Head.
        for head_idx in range(params.attn_head_count):
            head_start_idx = params.attn_head_dim * head_idx
            head_end_idx = params.attn_head_dim * (head_idx + 1)

            q_head = query[head_start_idx:head_end_idx]
            k_head = [k[head_start_idx:head_end_idx] for k, _ in kv_cache[block_idx]]
            v_head = [v[head_start_idx:head_end_idx] for _, v in kv_cache[block_idx]]

            attn_logits = [vec_dot_product(q_head, k) / params.attn_head_dim**0.5 for k in k_head]
            attn_probs = softmax(attn_logits)
            for dim in range(params.attn_head_dim):
                v_at_dim = [v[dim] for v in v_head]
                attn_output.append(vec_dot_product(attn_probs, v_at_dim))

        x = linear(attn_output, params.w_transformer_attn_out[block_idx])
        x = vec_sum(x, x_residual)

        # MLP Block.
        x_residual = x
        x = rms_norm(x)
        x = linear(x, params.w_transformer_mlp_fc1[block_idx])
        x = [elem.relu() for elem in x]
        x = linear(x, params.w_transformer_mlp_fc2[block_idx])
        x = vec_sum(x, x_residual)

    x = rms_norm(x)
    x = linear(x, params.w_lm_head)

    return x
