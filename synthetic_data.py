"""
    Module for generating synthetic data for tests
"""

from random import randint

from optimization_problems import Volume, Pool

def randomVolume(limits):
    return Volume(randint(1, limits[0]), randint(1, limits[1]), randint(1, limits[2]))

def randomPool(limits):
    return Pool(randint(1, limits[0]), randint(1, limits[1]), randint(1, limits[2]))

def generateRandomData(n_volumes: int, n_pools: int, volume_limits, pool_limits):
    assert len(volume_limits) == 3
    assert len(pool_limits) == 3
    volumes = []
    pools = []
    for _ in range(n_volumes):
        volumes.append(randomVolume(volume_limits))
    for _ in range(n_pools):
        pools.append(randomPool(pool_limits))
    return volumes, pools
