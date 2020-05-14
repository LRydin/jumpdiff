# Based on 'kramersmoyal'. Identical test scheme

import numpy as np
from itertools import product

from JumpDiff.kernels import *

def test_kernels():
    for dim in [1]:
        edges = [np.linspace(-10, 10, 100000 // 10**dim, endpoint=True)] * dim
        mesh = np.asarray(list(product(*edges)))
        dx = (edges[0][1] - edges[0][0]) ** dim
        for kernel in [epanechnikov, gaussian, uniform, triagular]:
            for bw in [0.1, 0.3, 0.5, 1.0, 1.5, 2.0]:
                kernel_ = kernel(mesh, bw=bw).reshape(
                    *(edge.size for edge in edges))
                assert np.allclose(kernel_.sum() * dx, 1, atol=1e-2)
