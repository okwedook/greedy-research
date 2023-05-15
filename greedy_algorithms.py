"""
    Implementations of greedy algorithms
"""

import numpy as np

from spaces import Space
from optimization_problems import OptimizationProblem

def simpleGreedy(opt: OptimizationProblem, X: Space, x=None, render_path=False):
    x = np.array(x) if x else np.empty(X.shape)
    if render_path:
        path = [x]
    while True:
        V = X.getVicinity(x)
        V = [v for v in V if opt.inConstraint(v)]
        V.sort(key=opt.f, reverse=True)
        if V and opt.f(V[0]) > opt.f(x):
            x = V[0]
            if render_path:
                path.append(x)
        else:
            break
    if render_path:
        return x, path
    return x

def LEGSplit(opt: OptimizationProblem, V, x):
    L, E, G = [], [], []
    gx = opt.g(x)
    fx = opt.f(x)
    for v in V:
        if not opt.inConstraint(v):
            continue
        fv = opt.f(v)
        if fv < fx:
            continue
        gv = opt.g(v)
        if gx > gv:
            L.append(v)
        elif fv > fx:
            if gx == gv:
                E.append(v)
            else:
                G.append(v)

    L.sort(key=lambda x: (fx - opt.f(x)) / (gx - opt.g(x)), reverse=True)
    E.sort(key=opt.f, reverse=True)
    G.sort(key=lambda x: (fx - opt.f(x)) / (gx - opt.g(x)), reverse=True)

    return L, E, G


def powerGreedy(opt: OptimizationProblem, X: Space, x=None, render_path=False):
    x = np.array(x) if x else np.empty(X.shape)

    if render_path:
        path = [x]

    while True:
        V = X.getVicinity(x)
        L, E, G = LEGSplit(opt, V, x)
        if L:
            x = L[0]
        elif E:
            x = E[0]
        elif G:
            x = G[0]
        else:
            break

        if render_path:
            path.append(x)

    if render_path:
        return x, path
    return x
