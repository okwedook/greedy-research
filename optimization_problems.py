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

class CloudResource:
    def __init__(self, vs, ps, bw):
        self.vs = vs
        self.ps = ps
        self.bw = bw

    def add(self, resource):
        self.vs += resource.vs
        self.ps += resource.ps
        self.bw += resource.bw

    def __repr__(self) -> str:
        return f"({self.vs} {self.ps} {self.bw})"

class Volume(CloudResource):
    pass

class Pool(CloudResource):
    pass

class CloudStorageScheduling(OptimizationProblem):
    def __init__(self, volumes, pools, thresholds):
        self.volumes = volumes
        self.pools = pools
        assert len(thresholds) == 3
        self.VST, self.PST, self.BWT = thresholds

    def getSpace(self) -> spaces.MatrixBinary:
        return spaces.MatrixBinary((len(self.volumes), len(self.pools)))

    def getBal(self, usage, resource, threshold):
        current = usage / resource
        return (threshold - current) / threshold

    def getBalResources(self, usage, pool):
        return [
            self.getBal(usage.vs, pool.vs, self.VST),
            self.getBal(usage.ps, pool.ps, self.PST),
            self.getBal(usage.bw, pool.bw, self.BWT),
        ]

    def getUsedResources(self, x):
        used_resources = []
        for _ in self.pools:
            used_resources.append(Pool(0, 0, 0))

        used_volumes = 0
        multiple_decision = 0

        for volume_resource, decision in zip(self.volumes, x):
            indices = []
            for i, d in enumerate(decision):
                if d:
                    indices.append(i)
            if not indices:
                continue
            if len(indices) > 1:
                multiple_decision += 1
            used_volumes += 1
            used_resources[indices[0]].add(volume_resource)

        return used_resources, used_volumes, multiple_decision

    def getStats(self, used_resources):
        good_stat = 0
        bad_stat = 0

        for usage, pool in zip(used_resources, self.pools):
            stat = self.getBalResources(usage, pool)
            min_stat = min(stat)
            if min_stat < 0:
                bad_stat += min_stat
            else:
                good_stat += min_stat

        return good_stat, bad_stat

    def f(self, x, add_volumes=True):
        assert len(self.volumes) == len(x)

        used_resources, used_volumes, _ = self.getUsedResources(x)

        good_stat, bad_stat = self.getStats(used_resources)

        if bad_stat != 0:
            return bad_stat

        return good_stat + (2 * used_volumes if add_volumes else 0)

    def g(self, x):
        assert len(self.volumes) == len(x)

        used_resources, _, multiple_decision = self.getUsedResources(x)

        if multiple_decision != 0:
            return multiple_decision

        good_stat, bad_stat = self.getStats(used_resources)

        if bad_stat:
            return -bad_stat

        return -good_stat
