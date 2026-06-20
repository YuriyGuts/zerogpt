"""Vector and matrix operations over autograd nodes."""

from zerogpt.autograd import AutoGradNode
from zerogpt.autograd import Matrix
from zerogpt.autograd import Vector


def vec_sum(a: Vector, b: Vector) -> Vector:
    """Add two vectors element-wise."""
    return [a_i + b_i for a_i, b_i in zip(a, b, strict=True)]


def rms_norm(vec: Vector, epsilon: float = 1e-5) -> Vector:
    """Normalize a vector with RMSNorm."""
    sum_squares = AutoGradNode.sum(elem * elem for elem in vec)
    denominator = (sum_squares / len(vec) + epsilon) ** 0.5
    return [elem / denominator for elem in vec]


def vec_dot_product(a: Vector, b: Vector) -> AutoGradNode:
    """Compute the dot product of two vectors (an n-ary operator for efficiency)."""
    total = 0.0
    children = []
    grads_wrt_children = []

    # f = a1 * b1 + a2 * b2 + ...
    # df/da1 = b1
    # df/db1 = a1
    # df/da2 = b2
    # df/db2 = a2

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
    """Multiply a matrix by a vector."""
    return [vec_dot_product(row, vec) for row in mat]


def linear(vec: Vector, mat: Matrix) -> Vector:
    """Apply a linear transformation to a vector, as defined by a transformation matrix."""
    return mat_vec_product(mat, vec)


def softmax(vec: Vector) -> Vector:
    """Compute the softmax of a vector (numerically stable version)."""
    max_value = max(elem.value for elem in vec)
    shifted = [elem - max_value for elem in vec]
    exps = [elem.exp() for elem in shifted]
    sum_exps = AutoGradNode.sum(elem for elem in exps)
    return [elem / sum_exps for elem in exps]


def log_softmax(vec: Vector) -> Vector:
    """Compute the log-softmax of a vector (numerically stable version)."""
    max_value = max(elem.value for elem in vec)
    shifted = [elem - max_value for elem in vec]
    log_sum_exps = AutoGradNode.sum(elem.exp() for elem in shifted).log()
    return [elem - log_sum_exps for elem in shifted]
