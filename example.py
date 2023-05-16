"""
    Example of library usage
"""

import numpy as np

from spaces import Box
from optimization_problems import QuadraticOptimization
from random_algorithms import bestSample, randomVicinity

X = Box([0, 0], [20, 20], dtype=np.int32)

qo = QuadraticOptimization([-1, -1], [40, 50], -349, lambda x: x[0] + x[1])

best_point_sample = bestSample(qo, X, n_tries=20)
random_vicinity_sample = randomVicinity(qo, X, [0, 0])

print(best_point_sample, qo.f(best_point_sample))
print(random_vicinity_sample, qo.f(random_vicinity_sample))
