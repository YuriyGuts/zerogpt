from dataclasses import dataclass

from zerogpt.autograd import Matrix
from zerogpt.autograd import Vector
from zerogpt.ops import linear
from zerogpt.ops import rms_norm
from zerogpt.ops import vec_sum


@dataclass(slots=True)
class GPTParams:
    @property
    def transformer_block_count(self) -> int:
        return 0

    w_token_emb: Matrix
    w_position_emb: Matrix

    w_transformer_attn_q: list[Matrix]
    w_transformer_attn_k: list[Matrix]
    w_transformer_attn_v: list[Matrix]


def gpt(
    token_id: int,
    position_id: int,
    params: GPTParams,
) -> Vector:
    token_emb = params.w_token_emb[token_id]
    position_emb = params.w_position_emb[position_id]
    x = vec_sum(token_emb, position_emb)

    for block_idx in range(params.transformer_block_count):
        x = rms_norm(x)
        q = linear(x, params.w_transformer_attn_q[block_idx])
        k = linear(x, params.w_transformer_attn_k[block_idx])
        v = linear(x, params.w_transformer_attn_v[block_idx])

        # TODO: to be continued...

    return x
