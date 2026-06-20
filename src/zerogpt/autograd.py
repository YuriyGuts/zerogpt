"""A simple reverse-mode automatic differentiation engine based on operator overloading."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

_TOPO_UNVISITED = 0
_TOPO_IN_PROGRESS = 1
_TOPO_FINISHED = 2


@dataclass(slots=True)
class AutoGradNode:
    """
    Represents a single scalar value in an autograd computation graph.

    When a node interacts with other operands, it builds up the graph under the hood.
    """

    # The value of the node.
    value: float

    # The gradient accumulated in the node.
    grad: float = 0.0

    # The nodes this value was computed from.
    children: tuple[AutoGradNode, ...] = ()

    # The local derivative of this value with respect to each child.
    grads_wrt_children: tuple[float, ...] = ()

    # Temporary state used while topologically sorting the graph.
    topo_state: int = _TOPO_UNVISITED

    def __add__(self, other: float | AutoGradNode) -> AutoGradNode:
        # f = a + const
        # df/da = 1
        if not isinstance(other, AutoGradNode):
            return AutoGradNode(
                value=self.value + other,
                children=(self,),
                grads_wrt_children=(1.0,),
            )

        # f = a + b
        # df/da = 1
        # df/db = 1
        return AutoGradNode(
            value=self.value + other.value,
            children=(self, other),
            grads_wrt_children=(1.0, 1.0),
        )

    def __radd__(self, other: float) -> AutoGradNode:
        # f = 2 + a
        # df/da = 1
        return self + other

    def __sub__(self, other: float | AutoGradNode) -> AutoGradNode:
        # f = a - const
        # df/da = 1
        if not isinstance(other, AutoGradNode):
            return AutoGradNode(
                value=self.value - other,
                children=(self,),
                grads_wrt_children=(1.0,),
            )

        # f = a - b
        # df/da = 1
        # df/db = -1
        return AutoGradNode(
            value=self.value - other.value,
            children=(self, other),
            grads_wrt_children=(1.0, -1.0),
        )

    def __rsub__(self, other: float) -> AutoGradNode:
        # f = 2 - a
        # df/da = -1
        return AutoGradNode(
            value=other - self.value,
            children=(self,),
            grads_wrt_children=(-1.0,),
        )

    def __mul__(self, other: float | AutoGradNode) -> AutoGradNode:
        # f = a * const
        # df/da = const
        if not isinstance(other, AutoGradNode):
            return AutoGradNode(
                value=self.value * other,
                children=(self,),
                grads_wrt_children=(other,),
            )

        # f = a * b
        # df/da = b
        # df/db = a
        return AutoGradNode(
            value=self.value * other.value,
            children=(self, other),
            grads_wrt_children=(other.value, self.value),
        )

    def __rmul__(self, other: float) -> AutoGradNode:
        # f = const * a
        # df/da = const
        return self * other

    def __truediv__(self, other: float | AutoGradNode) -> AutoGradNode:
        # f = a / const
        # df/da = 1 / const
        if not isinstance(other, AutoGradNode):
            return AutoGradNode(
                value=self.value / other,
                children=(self,),
                grads_wrt_children=(1.0 / other,),
            )

        # f = a / b
        # df/da = 1 / b
        # df/db = -a / b**2
        return AutoGradNode(
            value=self.value / other.value,
            children=(self, other),
            grads_wrt_children=(
                1.0 / other.value,
                -self.value / other.value**2,
            ),
        )

    def __rtruediv__(self, other: float) -> AutoGradNode:
        # f = const / a
        # df/da = -const / a ** 2
        return AutoGradNode(
            value=other / self.value,
            children=(self,),
            grads_wrt_children=(-other / self.value**2,),
        )

    def __pow__(self, other: float | AutoGradNode) -> AutoGradNode:
        # f = a ** const
        # df/da = const * a ** (const - 1)
        if not isinstance(other, AutoGradNode):
            return AutoGradNode(
                value=self.value**other,
                children=(self,),
                grads_wrt_children=(other * self.value ** (other - 1),),
            )

        # f = a ** b
        # df/da = b * a ** (b - 1)
        # df/db = a ** b * ln(a)
        value = self.value**other.value
        return AutoGradNode(
            value=value,
            children=(self, other),
            grads_wrt_children=(
                other.value * self.value ** (other.value - 1),
                value * math.log(self.value),
            ),
        )

    def __rpow__(self, other: float) -> AutoGradNode:
        # f = const ** a
        # df/da = const ** a * ln(const)
        return AutoGradNode(
            value=other**self.value,
            children=(self,),
            grads_wrt_children=(other**self.value * math.log(other),),
        )

    def __neg__(self) -> AutoGradNode:
        # f = -a
        # df/da = -1
        return AutoGradNode(
            value=-self.value,
            children=(self,),
            grads_wrt_children=(-1.0,),
        )

    def __format__(self, format_spec: str) -> str:
        return self.value.__format__(format_spec)

    def log(self) -> AutoGradNode:
        """Return the natural logarithm as a new node."""
        # f = ln a
        # df/da = 1 / a
        return AutoGradNode(
            value=math.log(self.value),
            children=(self,),
            grads_wrt_children=(1.0 / self.value,),
        )

    def exp(self) -> AutoGradNode:
        """Return e raised to this value as a new node."""
        # f = e ** a
        # df/da = e ** a
        value = math.exp(self.value)
        return AutoGradNode(
            value=value,
            children=(self,),
            grads_wrt_children=(value,),
        )

    def relu(self) -> AutoGradNode:
        """Return the ReLU activation as a new node."""
        # f = max(0, a)
        # df/da = 1 if a > 0 else 0
        return AutoGradNode(
            value=max(0.0, self.value),
            children=(self,),
            grads_wrt_children=(1.0 if self.value > 0 else 0.0,),
        )

    @classmethod
    def sum(cls, gen: Iterable[AutoGradNode]) -> AutoGradNode:
        """Sum many nodes into a single node (an n-ary operator for efficiency)."""
        # f = a + b + c + d + ...
        # df/da = 1
        # df/db = 1
        # ...
        items = tuple(gen)
        return AutoGradNode(
            value=sum(item.value for item in items),
            children=items,
            grads_wrt_children=(1.0,) * len(items),
        )

    def backpropagate(self) -> None:
        """Propagate the gradient from this node back to every child node."""
        self.grad = 1.0
        topo_order = topological_sort(self)
        for node in reversed(topo_order):
            for child, grad_wrt_child in zip(node.children, node.grads_wrt_children, strict=True):
                child.grad += node.grad * grad_wrt_child


def topological_sort(final_node: AutoGradNode) -> list[AutoGradNode]:
    """Return all nodes feeding into `final_node`, in topological order."""
    result = []

    # Use an explicit stack instead of recursion to avoid hitting the system recursion limit.
    stack = deque()
    final_node.topo_state = _TOPO_IN_PROGRESS
    stack.append((final_node, iter(final_node.children)))

    while stack:
        current_node, unprocessed_children = stack[-1]
        child = next(unprocessed_children, None)
        if child is None:
            current_node.topo_state = _TOPO_FINISHED
            result.append(current_node)
            stack.pop()
            continue

        if child.topo_state == _TOPO_IN_PROGRESS:
            raise RuntimeError(f"Cycle detected in {child} via {current_node}")
        elif child.topo_state == _TOPO_UNVISITED:
            child.topo_state = _TOPO_IN_PROGRESS
            stack.append((child, iter(child.children)))

    for node in result:
        node.topo_state = _TOPO_UNVISITED

    return result


Vector = list[AutoGradNode]
Matrix = list[Vector]
