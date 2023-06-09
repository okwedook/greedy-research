"""
    Description of different spaces, which include:
    - Discrete
    - Box (Euclidean N-dimensional space)
    - Other
"""

from typing import Optional, Tuple, Sequence, Generic, TypeVar
from itertools import product
from copy import deepcopy

import numpy as np


T_cov_co = TypeVar("T_cov_co", covariant=True)

class Space(Generic[T_cov_co]):
    def __init__(
        self,
        shape: Sequence[int],
        dtype: np.dtype = np.float32,
        seed: Optional[int] = None
    ):
        self.shape = shape
        self.dtype = dtype
        seed = seed if seed else 0
        self.rng = np.random.default_rng(seed)

    def genSample(self) -> any:
        raise NotImplementedError

    def getShape(self) -> Optional[Tuple[int, ...]]:
        return self.shape

# Closed box in Euclidean space
class Box(Space[np.dtype]):
    def __init__(
        self,
        low: np.ndarray,
        high: np.ndarray,
        dtype: np.dtype = np.float32,
        seed: Optional[int] = None
    ):
        low = np.array(low)
        high = np.array(high)
        assert low.shape == high.shape
        super().__init__(low.shape, dtype, seed)
        if np.dtype(dtype).kind == "i":
            low = np.ceil(low)
            high = np.floor(high) + 1
        self.low = low
        self.high = high

    def genSample(self) -> np.ndarray:
        sample = self.rng.uniform(low=self.low, high=self.high)
        if np.dtype(self.dtype).kind == "i":
            sample = np.floor(sample)

        return sample.astype(self.dtype)

    def inside(self, x):
        return np.all(np.less_equal(self.low, x)) and np.all(np.greater_equal(self.high, x))

    def getVicinity(self, x):
        if np.dtype(self.dtype).kind != "i":
            raise NotImplementedError

        x = np.array(x)

        ans = []

        for d in product([-1, 0, 1], repeat=len(self.low)):
            if self.inside(x + d):
                ans.append(x + d)

        return ans

class MatrixBinary(Space):
    def __init__(
        self,
        shape: Sequence[int],
        seed: Optional[int] = None
    ):
        super().__init__(shape, 'bool', seed)

    def genSample(self):
        matrix = self.rng.random(self.shape)
        matrix = np.round(matrix).astype('bool')
        return matrix

    def getVicinity(self, x, cloudVicinity=True):
        assert x.shape == self.shape

        V = []

        for i in range(self.shape[0]):
            if cloudVicinity and np.any(x[i]):
                continue
            for j in range(self.shape[1]):
                x_copy = deepcopy(x)
                x_copy[i][j] = not x_copy[i][j]
                V.append(x_copy)

        return V
