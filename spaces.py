from random import randint
from typing import Optional, Tuple, Sequence, Union, Type, Generic, TypeVar
import numpy as np

T_cov = TypeVar("T_cov", covariant=True)

class Space(Generic[T_cov]):
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

# Creates a space with n options in range [0, n)
class Discrete(Space):
    def __init__(self, n) -> None:
        self.n = n

    def genSample(self) -> int:
        return randint(0, self.n - 1)

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
        assert(low.shape == high.shape)
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
        
