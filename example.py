from spaces import Box
from optimization_problems import QuadraticOptimization
from random_algorithms import bestSample
import numpy as np

X = Box([0, 0], [20, 20], dtype=np.int32)

qo = QuadraticOptimization([1, 1], [-40, -50], 349, lambda x: x[0] + x[1])

best_point, best_value = bestSample(qo, X, n_tries=10000)

print(best_point, best_value)