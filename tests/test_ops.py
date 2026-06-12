import math

import pytest

from tests.helpers import maybe_import

AutoGradNode = maybe_import("zerogpt.autograd", "AutoGradNode")
linear = maybe_import("zerogpt.ops", "linear")
log_softmax = maybe_import("zerogpt.ops", "log_softmax")
mat_vec_product = maybe_import("zerogpt.ops", "mat_vec_product")
rms_norm = maybe_import("zerogpt.ops", "rms_norm")
softmax = maybe_import("zerogpt.ops", "softmax")
vec_dot_product = maybe_import("zerogpt.ops", "vec_dot_product")
vec_sum = maybe_import("zerogpt.ops", "vec_sum")


def _vec(values):
    return [AutoGradNode(v) for v in values]


def _mat(rows):
    return [_vec(row) for row in rows]


def _assert_grads_match_numeric(build_loss, leaves, eps=1e-6):
    # Compare backprop gradients against a central finite-difference estimate.
    for leaf in leaves:
        leaf.grad = 0.0
    build_loss().backpropagate()

    for leaf in leaves:
        analytic = leaf.grad
        original = leaf.value
        leaf.value = original + eps
        loss_plus = build_loss().value
        leaf.value = original - eps
        loss_minus = build_loss().value
        leaf.value = original
        numeric = (loss_plus - loss_minus) / (2 * eps)
        assert analytic == pytest.approx(numeric, abs=1e-4)


def test_vec_sum_elementwise_addition():
    # GIVEN two vectors of equal length
    a = _vec([1.0, 2.0, 3.0])
    b = _vec([10.0, 20.0, 30.0])

    # WHEN summing them element-wise
    result = vec_sum(a, b)

    # THEN each output element equals the sum of the inputs at that index
    assert [r.value for r in result] == [11.0, 22.0, 33.0]


def test_vec_sum_length_mismatch_raises():
    # GIVEN two vectors of different lengths
    a = _vec([1.0, 2.0])
    b = _vec([1.0, 2.0, 3.0])

    # WHEN attempting to sum them
    # THEN `zip(strict=True)` raises `ValueError`
    with pytest.raises(ValueError, match="longer than"):
        vec_sum(a, b)


def test_vec_dot_product_value():
    # GIVEN two vectors of equal length
    a = _vec([1.0, 2.0, 3.0])
    b = _vec([4.0, 5.0, 6.0])

    # WHEN computing the dot product
    result = vec_dot_product(a, b)

    # THEN the result equals the sum of element-wise products
    assert result.value == 32.0


def test_vec_dot_product_length_mismatch_raises():
    # GIVEN two vectors of different lengths
    a = _vec([1.0, 2.0])
    b = _vec([1.0])

    # WHEN attempting to compute the dot product
    # THEN `zip(strict=True)` raises `ValueError`
    with pytest.raises(ValueError, match=" than "):
        vec_dot_product(a, b)


def test_mat_vec_product_shape_and_values():
    # GIVEN a 3x2 matrix and a length-2 vector
    mat = _mat([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    vec = _vec([7.0, 8.0])

    # WHEN multiplying the matrix by the vector
    result = mat_vec_product(mat, vec)

    # THEN the result is a length-3 vector of per-row dot products
    assert [r.value for r in result] == [23.0, 53.0, 83.0]


def test_linear_matches_mat_vec_product():
    # GIVEN a matrix and a vector
    mat = _mat([[1.0, 2.0], [3.0, 4.0]])
    vec = _vec([1.0, 1.0])

    # WHEN calling `linear(vec, mat)`
    result = linear(vec, mat)

    # THEN it is equivalent to `mat_vec_product(mat, vec)` with arguments swapped
    assert [r.value for r in result] == [3.0, 7.0]


def test_softmax_outputs_sum_to_one():
    # GIVEN an arbitrary input vector
    vec = _vec([1.0, 2.0, 3.0])

    # WHEN applying softmax
    result = softmax(vec)

    # THEN the output is a probability distribution that sums to 1
    assert sum(r.value for r in result) == pytest.approx(1.0)


def test_softmax_all_outputs_positive():
    # GIVEN an input mixing negative, zero, and positive values
    vec = _vec([-5.0, 0.0, 10.0])

    # WHEN applying softmax
    result = softmax(vec)

    # THEN every output is strictly positive
    assert all(r.value > 0 for r in result)


def test_softmax_uniform_input_yields_uniform_output():
    # GIVEN an input vector where every element is identical
    vec = _vec([5.0, 5.0, 5.0, 5.0])

    # WHEN applying softmax
    result = softmax(vec)

    # THEN the output is the uniform distribution
    for r in result:
        assert r.value == pytest.approx(0.25)


def test_softmax_is_numerically_stable_for_large_inputs():
    # GIVEN an input whose naive `exp()` would overflow
    vec = _vec([1000.0, 1001.0, 1002.0])

    # WHEN applying softmax (which subtracts the max for stability)
    result = softmax(vec)

    # THEN every element is finite and the distribution still sums to 1
    assert sum(r.value for r in result) == pytest.approx(1.0)
    assert all(math.isfinite(r.value) for r in result)


def test_softmax_exact_values_for_known_input():
    # GIVEN a small input vector with well-known canonical softmax outputs
    vec = _vec([1.0, 2.0, 3.0])

    # WHEN applying softmax
    result = softmax(vec)

    # THEN each output equals `exp(x_i) / sum(exp(x_j))`
    expected = [
        0.09003057317038046,
        0.24472847105479764,
        0.6652409557748219,
    ]
    for r, e in zip(result, expected, strict=True):
        assert r.value == pytest.approx(e)


def test_softmax_of_log_counts_yields_proportional_distribution():
    # GIVEN inputs that are the logs of integer counts `[1, 2, 3, 4]`,
    # softmax should recover the normalized counts: `[1/10, 2/10, 3/10, 4/10]`.
    vec = _vec([math.log(1), math.log(2), math.log(3), math.log(4)])

    # WHEN applying softmax
    result = softmax(vec)

    # THEN the result is the rational distribution `count_i / sum(counts)`
    expected = [0.1, 0.2, 0.3, 0.4]
    for r, e in zip(result, expected, strict=True):
        assert r.value == pytest.approx(e)


def test_log_softmax_exp_matches_softmax():
    # GIVEN the same input vector used twice
    vec_for_log = _vec([1.0, 2.0, 3.0])
    vec_for_direct = _vec([1.0, 2.0, 3.0])

    # WHEN computing both `log_softmax` and `softmax`
    log_probs = log_softmax(vec_for_log)
    direct_probs = softmax(vec_for_direct)

    # THEN `exp(log_softmax(x)) == softmax(x)` element-wise
    for lp, dp in zip(log_probs, direct_probs, strict=True):
        assert math.exp(lp.value) == pytest.approx(dp.value)


def test_log_softmax_is_numerically_stable_for_large_inputs():
    # GIVEN an input whose naive `exp()` would overflow
    vec = _vec([1000.0, 1001.0, 1002.0])

    # WHEN applying log_softmax
    log_probs = log_softmax(vec)

    # THEN every output element is finite
    assert all(math.isfinite(p.value) for p in log_probs)


def test_log_softmax_exact_values_for_known_input():
    # GIVEN a small input vector
    vec = _vec([1.0, 2.0, 3.0])

    # WHEN applying log_softmax
    result = log_softmax(vec)

    # THEN each output equals `x_i - log(sum_j exp(x_j))`
    log_sum_exp = math.log(math.exp(1.0) + math.exp(2.0) + math.exp(3.0))
    expected = [1.0 - log_sum_exp, 2.0 - log_sum_exp, 3.0 - log_sum_exp]
    for r, e in zip(result, expected, strict=True):
        assert r.value == pytest.approx(e)


def test_rms_norm_preserves_vector_length():
    # GIVEN a vector
    vec = _vec([1.0, 2.0, 3.0, 4.0])

    # WHEN applying RMS norm
    result = rms_norm(vec)

    # THEN the output has the same dimensionality
    assert len(result) == len(vec)


def test_rms_norm_scales_to_unit_rms():
    # GIVEN a vector with non-trivial RMS
    vec = _vec([3.0, 4.0])

    # WHEN normalizing with epsilon=0
    result = rms_norm(vec, epsilon=0.0)

    # THEN the output has RMS exactly 1
    rms_after = math.sqrt(sum(r.value**2 for r in result) / len(result))
    assert rms_after == pytest.approx(1.0)


def test_rms_norm_unit_vector_approximately_unchanged():
    # GIVEN a vector that already has unit RMS
    vec = _vec([1.0, 1.0, 1.0, 1.0])

    # WHEN normalizing with the default epsilon
    result = rms_norm(vec)

    # THEN the output is essentially unchanged (epsilon shrinks the scaler very slightly)
    for r in result:
        assert r.value == pytest.approx(1.0, rel=1e-3)


def test_rms_norm_exact_values_for_known_input():
    # GIVEN a vector whose per-element output can be derived by hand
    vec = _vec([1.0, 2.0, 3.0, 4.0])

    # WHEN normalizing with epsilon=0
    result = rms_norm(vec, epsilon=0.0)

    # THEN each output equals `x_i / sqrt(mean(x_j^2))`
    mean_squares = (1.0**2 + 2.0**2 + 3.0**2 + 4.0**2) / 4
    rms = math.sqrt(mean_squares)
    expected = [1.0 / rms, 2.0 / rms, 3.0 / rms, 4.0 / rms]
    for r, e in zip(result, expected, strict=True):
        assert r.value == pytest.approx(e)


def test_log_softmax_gradients_match_numeric():
    # GIVEN logits and the index of the "true" next token
    logits = _vec([0.3, -0.7, 1.1, 0.0])
    true_index = 2

    # WHEN the loss is the negative log-probability of the true token
    def build_loss():
        return -log_softmax(logits)[true_index]

    # THEN backprop gradients match finite differences
    _assert_grads_match_numeric(build_loss, logits)


def test_softmax_gradients_match_numeric():
    # GIVEN softmax inputs and fixed weights that collapse the output to a scalar
    inputs = _vec([0.5, -0.2, 0.9])
    fixed_values = _vec([1.0, 2.0, -1.0])

    # WHEN the loss is a weighted sum of the softmax probabilities
    def build_loss():
        return vec_dot_product(softmax(inputs), fixed_values)

    # THEN backprop gradients match finite differences
    _assert_grads_match_numeric(build_loss, inputs)


def test_rms_norm_gradients_match_numeric():
    # GIVEN an input vector
    vec = _vec([0.4, -1.2, 0.7, 2.0])

    # WHEN the loss is the sum of the normalized components
    def build_loss():
        return AutoGradNode.sum(rms_norm(vec))

    # THEN backprop gradients match finite differences
    _assert_grads_match_numeric(build_loss, vec)
