"""
    An optimization problem requires:
    - Parameter space X
    - Objective function f -> R
    - Constraint function g -> R ^ k
"""

import numpy as np

import spaces

class OptimizationProblem:
    def __init__(self) -> None:
        raise NotImplementedError

    def getSpace(self) -> spaces.Space:
        raise NotImplementedError

    def eval(self, x):
        return self.f(x)

    def getConstraint(self, x):
        return self.g(x)

    def inConstraint(self, x) -> bool:
        return self.g(x) <= 0

# ax^2 + by^2 + ... + c >= 0
class QuadraticOptimization(OptimizationProblem):
    def __init__(self, c2: np.array, c1: np.array, c0: np.float32, f) -> None:
        assert(len(c2) == len(c1))
        self.c2 = c2
        self.c1 = c1
        self.c0 = c0
        self.f = f

    def getSpace(self) -> spaces.Box:
        return spaces.Box([-np.inf, -np.inf], [np.inf, np.inf])

    def g(self, x: np.array):
        return np.dot(np.multiply(x, x), self.c2) + np.dot(x, self.c1) + self.c0

# ax + by + ... + c >= 0
# class LinearOptimization(OptimizationProblem):
#     def __init__(self) -> None:
#         super().__init__()
