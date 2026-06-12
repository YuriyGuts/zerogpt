from zerogpt.autograd import AutoGradNode
from zerogpt.autograd import Matrix
from zerogpt.autograd import Vector


def vec_sum(a: Vector, b: Vector) -> Vector:
    return [a_i + b_i for a_i, b_i in zip(a, b, strict=True)]


def rms_norm(vec: Vector, epsilon: float = 1e-5) -> Vector:
    sum_squares = AutoGradNode.sum(elem * elem for elem in vec)
    denominator = (sum_squares / len(vec) + epsilon) ** 0.5
    return [elem / denominator for elem in vec]


def vec_dot_product(a: Vector, b: Vector) -> AutoGradNode:
    total = 0.0
    children = []
    grads_wrt_children = []

    # f = a * b
    # df/da = b
    # df/db = a

    for a_i, b_i in zip(a, b, strict=True):
        total += a_i.value * b_i.value
        children.append(a_i)
        children.append(b_i)
        grads_wrt_children.append(b_i.value)
        grads_wrt_children.append(a_i.value)

    return AutoGradNode(
        value=total,
        children=tuple(children),
        grads_wrt_children=tuple(grads_wrt_children),
    )


def mat_vec_product(mat: Matrix, vec: Vector) -> Vector:
    return [vec_dot_product(row, vec) for row in mat]


def linear(vec: Vector, mat: Matrix) -> Vector:
    return mat_vec_product(mat, vec)
