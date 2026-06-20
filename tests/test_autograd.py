import math

import pytest

from zerogpt.autograd import AutoGradNode
from zerogpt.autograd import topological_sort


def test_autograd_node_default_grad_and_no_children():
    # GIVEN a node created with only a value
    node = AutoGradNode(5.0)

    # WHEN inspecting its uninitialized fields
    grad = node.grad
    children = node.children

    # THEN grad starts at 0 and there are no upstream children
    assert node.value == 5.0
    assert grad == 0.0
    assert children == ()


def test_neg_value_and_gradient():
    # GIVEN a leaf node
    a = AutoGradNode(4.0)

    # WHEN negating it and backpropagating
    y = -a
    y.backpropagate()

    # THEN the value flips sign and the input receives grad -1
    assert y.value == -4.0
    assert a.grad == -1.0


def test_add_two_nodes_value_and_gradients():
    # GIVEN two leaf nodes
    a = AutoGradNode(2.0)
    b = AutoGradNode(3.0)

    # WHEN summing them and backpropagating
    y = a + b
    y.backpropagate()

    # THEN the value is the sum and both inputs receive grad 1
    assert y.value == 5.0
    assert a.grad == 1.0
    assert b.grad == 1.0


def test_add_node_and_constant():
    # GIVEN a leaf node
    a = AutoGradNode(2.0)

    # WHEN adding a plain int and backpropagating
    y = a + 5
    y.backpropagate()

    # THEN the constant participates in the value but only `a` gets a grad
    assert y.value == 7.0
    assert a.grad == 1.0


def test_radd_constant_plus_node():
    # GIVEN a leaf node
    a = AutoGradNode(2.0)

    # WHEN the constant appears on the left of `+`
    y = 5 + a
    y.backpropagate()

    # THEN `__radd__` is invoked and the result matches `a + 5`
    assert y.value == 7.0
    assert a.grad == 1.0


def test_mul_two_nodes_value_and_gradients():
    # GIVEN two leaf nodes
    a = AutoGradNode(3.0)
    b = AutoGradNode(4.0)

    # WHEN multiplying them and backpropagating
    y = a * b
    y.backpropagate()

    # THEN dy/da = b and dy/db = a
    assert y.value == 12.0
    assert a.grad == 4.0
    assert b.grad == 3.0


def test_rmul_constant_times_node():
    # GIVEN a leaf node
    a = AutoGradNode(3.0)

    # WHEN the constant appears on the left of `*`
    y = 4 * a
    y.backpropagate()

    # THEN `__rmul__` is invoked and the gradient equals the constant
    assert y.value == 12.0
    assert a.grad == 4.0


def test_sub_value_and_gradients():
    # GIVEN two leaf nodes
    a = AutoGradNode(10.0)
    b = AutoGradNode(3.0)

    # WHEN subtracting them and backpropagating
    y = a - b
    y.backpropagate()

    # THEN the minuend receives grad +1 and the subtrahend grad -1
    assert y.value == 7.0
    assert a.grad == 1.0
    assert b.grad == -1.0


def test_sub_node_and_constant():
    # GIVEN a leaf node
    a = AutoGradNode(2.0)

    # WHEN subtracting a plain int and backpropagating
    y = a - 5
    y.backpropagate()

    # THEN the constant participates in the value but `a` still receives grad +1
    assert y.value == -3.0
    assert a.grad == 1.0


def test_rsub_constant_minus_node():
    # GIVEN a leaf node
    a = AutoGradNode(3.0)

    # WHEN the constant appears on the left of `-`
    y = 10 - a
    y.backpropagate()

    # THEN `__rsub__` is invoked and the gradient is -1
    assert y.value == 7.0
    assert a.grad == -1.0


def test_truediv_value_and_gradients():
    # GIVEN two leaf nodes
    a = AutoGradNode(8.0)
    b = AutoGradNode(2.0)

    # WHEN dividing them and backpropagating
    y = a / b
    y.backpropagate()

    # THEN dy/da = 1/b and dy/db = -a/b^2
    assert y.value == pytest.approx(4.0)
    assert a.grad == pytest.approx(0.5)
    assert b.grad == pytest.approx(-2.0)


def test_rtruediv_constant_divided_by_node():
    # GIVEN a leaf node
    b = AutoGradNode(2.0)

    # WHEN the constant appears on the left of `/`
    y = 8 / b
    y.backpropagate()

    # THEN `__rtruediv__` is invoked and dy/db = -8/b^2
    assert y.value == pytest.approx(4.0)
    assert b.grad == pytest.approx(-2.0)


def test_pow_with_constant_exponent():
    # GIVEN a leaf node
    a = AutoGradNode(3.0)

    # WHEN raising to an integer power and backpropagating
    y = a**2
    y.backpropagate()

    # THEN dy/da = 2 * a
    assert y.value == 9.0
    assert a.grad == 6.0


def test_pow_with_autograd_exponent():
    # GIVEN a base and an exponent that are both nodes
    a = AutoGradNode(2.0)
    b = AutoGradNode(3.0)

    # WHEN raising base to exponent and backpropagating
    y = a**b
    y.backpropagate()

    # THEN dy/da = b * a^(b-1) and dy/db = a^b * ln(a)
    assert y.value == 8.0
    assert a.grad == pytest.approx(12.0)
    assert b.grad == pytest.approx(8.0 * math.log(2.0))


def test_rpow_constant_base_to_node_exponent():
    # GIVEN a leaf node used as an exponent
    a = AutoGradNode(3.0)

    # WHEN raising a constant to the node and backpropagating
    y = 2**a
    y.backpropagate()

    # THEN dy/da = 2^a * ln(2)
    assert y.value == 8.0
    assert a.grad == pytest.approx(8.0 * math.log(2.0))


def test_log_value_and_gradient():
    # GIVEN a positive leaf node
    a = AutoGradNode(2.0)

    # WHEN taking the natural log and backpropagating
    y = a.log()
    y.backpropagate()

    # THEN dy/da = 1/a
    assert y.value == pytest.approx(math.log(2.0))
    assert a.grad == pytest.approx(0.5)


def test_exp_value_and_gradient():
    # GIVEN a leaf node
    a = AutoGradNode(1.5)

    # WHEN exponentiating and backpropagating
    y = a.exp()
    y.backpropagate()

    # THEN dy/da = exp(a)
    assert y.value == pytest.approx(math.exp(1.5))
    assert a.grad == pytest.approx(math.exp(1.5))


def test_relu_positive_value_passes_through():
    # GIVEN a positive input
    a = AutoGradNode(3.0)

    # WHEN applying ReLU and backpropagating
    y = a.relu()
    y.backpropagate()

    # THEN the value is unchanged and the gradient is 1
    assert y.value == 3.0
    assert a.grad == 1.0


def test_relu_negative_value_zeroed():
    # GIVEN a negative input
    a = AutoGradNode(-2.0)

    # WHEN applying ReLU and backpropagating
    y = a.relu()
    y.backpropagate()

    # THEN the value is clamped to 0 and the gradient is 0
    assert y.value == 0.0
    assert a.grad == 0.0


def test_relu_zero_value_grad_zero():
    # GIVEN an input exactly at zero
    a = AutoGradNode(0.0)

    # WHEN applying ReLU and backpropagating
    y = a.relu()
    y.backpropagate()

    # THEN the gradient is zero because the implementation uses a strict `> 0` check
    assert y.value == 0.0
    assert a.grad == 0.0


def test_backpropagate_accumulates_grad_through_shared_subgraph():
    # GIVEN a leaf node referenced twice as a child of the same parent
    a = AutoGradNode(3.0)

    # WHEN computing y = a * a and backpropagating
    y = a * a
    y.backpropagate()

    # THEN the gradients along both edges accumulate: dy/da = 2*a = 6
    assert y.value == 9.0
    assert a.grad == 6.0


def test_backpropagate_composite_expression():
    # GIVEN three leaves
    a = AutoGradNode(1.0)
    b = AutoGradNode(2.0)
    c = AutoGradNode(4.0)

    # WHEN computing y = (a + b) * c and backpropagating
    y = (a + b) * c
    y.backpropagate()

    # THEN dy/da = c, dy/db = c, and dy/dc = a + b
    assert y.value == 12.0
    assert a.grad == 4.0
    assert b.grad == 4.0
    assert c.grad == 3.0


def test_sum_classmethod_aggregates_gradients():
    # GIVEN a few leaf nodes
    a = AutoGradNode(1.0)
    b = AutoGradNode(2.0)
    c = AutoGradNode(3.0)

    # WHEN reducing them with `AutoGradNode.sum` and backpropagating
    y = AutoGradNode.sum([a, b, c])
    y.backpropagate()

    # THEN every input contributes grad 1
    assert y.value == 6.0
    assert a.grad == 1.0
    assert b.grad == 1.0
    assert c.grad == 1.0


def test_format_delegates_to_underlying_float():
    # GIVEN a node holding a non-trivial float
    a = AutoGradNode(3.14159)

    # WHEN applying a `f"{...:.2f}"` format specifier
    formatted = f"{a:.2f}"

    # THEN it formats like the underlying float
    assert formatted == "3.14"


def test_topological_sort_places_root_last():
    # GIVEN a small DAG: y = a + b
    a = AutoGradNode(1.0)
    b = AutoGradNode(2.0)
    y = a + b

    # WHEN sorting topologically from the root
    topo = topological_sort(y)

    # THEN dependencies come first and the root comes last
    assert topo[-1] is y
    assert a in topo
    assert b in topo


def test_topological_sort_visits_each_node_once_with_shared_child():
    # GIVEN y = a * a, where the same child appears twice in y.children
    a = AutoGradNode(3.0)
    y = a * a

    # WHEN sorting topologically from the root
    topo = topological_sort(y)

    # THEN `a` appears only once because the algorithm dedupes by node id
    assert topo.count(a) == 1
    assert topo[-1] is y


def test_topological_sort_raises_on_cycle():
    # GIVEN two nodes wired into a cycle
    a = AutoGradNode(1.0)
    b = AutoGradNode(2.0, children=(a,), grads_wrt_children=(1.0,))
    a.children = (b,)
    a.grads_wrt_children = (1.0,)

    # WHEN running the sort
    # THEN a `RuntimeError` mentioning the cycle is raised
    with pytest.raises(RuntimeError, match="Cycle"):
        topological_sort(a)
