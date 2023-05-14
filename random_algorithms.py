from optimization_problems import OptimizationProblem
from spaces import Space
import numpy as np

def bestSample(opt: OptimizationProblem, X: Space, n_tries: int = 10):
    best_value = -np.inf
    best_point = None
    for t in range(n_tries):
        x = X.genSample()
        if opt.inConstraint(x):
            value = opt.f(x)
            if value > best_value:
                best_value = value
                best_point = x
    return best_point, best_value
