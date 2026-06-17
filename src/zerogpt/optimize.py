from collections.abc import Sequence

from zerogpt.autograd import AutoGradNode


class AdamOptimizer:
    def __init__(
        self,
        params: Sequence[AutoGradNode],
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.params = list(params)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.timestep_idx = 0
        self.m = [0.0] * len(self.params)
        self.v = [0.0] * len(self.params)

    def step(self) -> None:
        self.timestep_idx += 1
        for param_idx, param in enumerate(self.params):
            self.m[param_idx] = self.beta1 * self.m[param_idx] + (1.0 - self.beta1) * param.grad
            self.v[param_idx] = self.beta2 * self.v[param_idx] + (1.0 - self.beta2) * param.grad**2
            m_hat = self.m[param_idx] / (1.0 - self.beta1**self.timestep_idx)
            v_hat = self.v[param_idx] / (1.0 - self.beta2**self.timestep_idx)
            param.value -= self.learning_rate * m_hat / (v_hat**0.5 + self.epsilon)
            param.grad = 0.0
