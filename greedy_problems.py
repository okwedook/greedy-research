import numpy as np
import spaces

'''
An optimization requires:
- Parameter space X
- Objective function f -> R
- Constraint function g -> R ^ k
'''

class OptimizationProblem:
    def __init__(self) -> None:
        self.f = lambda x: 0
        self.g = lambda x: 0

    def getSpace(self) -> spaces.Space:
        raise NotImplementedError
    
    def eval(self, x):
        return self.f(x)
    
    def getConstraint(self, x):
        return self.g(x)

# ax^2 + by^2 + ... + c >= 0
class QuadraticOptimization(OptimizationProblem):
    def __init__(self, c: np.array, f) -> None:
        self.c = c
        self.f = f

    def getSpace(self) -> spaces.Box2D:
        return spaces.Box2D(len(self.c) - 1)

    def g(self, x: np.array):
        return np.dot(np.multiply(x, x), self.c[:-1]) + self.c[-1]

# ax + by + ... + c >= 0
class LinearOptimization(OptimizationProblem):
    def __init__(self) -> None:
        super().__init__()
