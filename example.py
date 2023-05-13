from spaces import Box
from greedy_problems import QuadraticOptimization
import numpy as np

X = Box([0, 0], [20, 20], dtype=np.int32)

qo = QuadraticOptimization([1, 1], [-40, -50], 349, lambda x: x[0] + x[1])

for _ in range(1000):
    x, y = X.genSample()
    if qo.g([x, y]) >= 0:
        print(x, y, qo.f([x, y]))