import pytest

from tests.helpers import maybe_import

AutoGradNode = maybe_import("zerogpt.autograd", "AutoGradNode")
AdamOptimizer = maybe_import("zerogpt.optimize", "AdamOptimizer")


def test_step_moves_param_opposite_to_positive_gradient():
    # GIVEN a parameter with a positive gradient
    p = AutoGradNode(1.0)
    p.grad = 0.5
    opt = AdamOptimizer([p], learning_rate=0.1, beta1=0.9, beta2=0.999)

    # WHEN running one optimization step
    opt.step()

    # THEN the parameter has decreased (descended along the gradient)
    assert p.value < 1.0


def test_step_moves_param_with_negative_gradient_upward():
    # GIVEN a parameter with a negative gradient
    p = AutoGradNode(1.0)
    p.grad = -0.5
    opt = AdamOptimizer([p], learning_rate=0.1, beta1=0.9, beta2=0.999)

    # WHEN running one optimization step
    opt.step()

    # THEN the parameter has increased
    assert p.value > 1.0


def test_step_resets_param_grad_to_zero():
    # GIVEN a parameter with a non-zero gradient
    p = AutoGradNode(1.0)
    p.grad = 0.5
    opt = AdamOptimizer([p], learning_rate=0.1, beta1=0.9, beta2=0.999)

    # WHEN running one step
    opt.step()

    # THEN the parameter's gradient has been zeroed for the next iteration
    assert p.grad == 0


def test_step_first_update_magnitude_equals_learning_rate():
    # GIVEN a parameter with a known gradient on the very first step
    p = AutoGradNode(10.0)
    p.grad = 2.0
    opt = AdamOptimizer([p], learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8)

    # WHEN running the first step
    opt.step()

    # THEN the update magnitude equals `lr * sign(grad)` (bias-corrected Adam at t=1)
    assert p.value == pytest.approx(10.0 - 0.1, rel=1e-3)


def test_step_no_update_when_grad_is_zero():
    # GIVEN a parameter whose gradient is exactly zero
    p = AutoGradNode(7.0)
    p.grad = 0.0
    opt = AdamOptimizer([p], learning_rate=0.1, beta1=0.9, beta2=0.999)

    # WHEN running one step
    opt.step()

    # THEN the parameter value is unchanged
    assert p.value == 7.0


def test_step_uses_independent_state_per_parameter():
    # GIVEN two parameters that share the same starting value and gradient
    p1 = AutoGradNode(5.0)
    p2 = AutoGradNode(5.0)
    p1.grad = 1.0
    p2.grad = 1.0
    opt = AdamOptimizer([p1, p2], learning_rate=0.1, beta1=0.9, beta2=0.999)

    # WHEN running one step
    opt.step()

    # THEN both parameters move identically because each has its own moment state
    assert p1.value == pytest.approx(p2.value)


def test_multiple_steps_continue_decreasing_param_with_constant_positive_grad():
    # GIVEN a fresh optimizer and a parameter
    p = AutoGradNode(10.0)
    opt = AdamOptimizer([p], learning_rate=0.1, beta1=0.9, beta2=0.999)

    # WHEN running several steps with a constant positive gradient
    history = []
    for _ in range(5):
        p.grad = 1.0
        opt.step()
        history.append(p.value)

    # THEN the parameter value strictly decreases step over step
    assert history == sorted(history, reverse=True)
