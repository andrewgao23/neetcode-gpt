import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        value = 0
        for i in range(len(x)):
            value += x[i] * w[i]
            i += 1
        value += b
        if activation == "sigmoid":
            return np.round((1/(1+np.exp(-1*value))),5)
        elif activation == "relu":
            if value > 0:
                return np.round(value,5)
            else:
                return 0.0
        else:
            return 0.0
        pass
