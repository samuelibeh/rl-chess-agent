import numpy as np
import pytest

from chessrl.encoding import POLICY_SIZE


class UniformEvaluator:
    """Deterministic stand-in for a network: uniform policy, zero value. Records batch sizes."""

    def __init__(self):
        self.batch_sizes = []
        self.calls = 0
        self.positions = 0

    def __call__(self, planes):
        n = len(planes)
        self.batch_sizes.append(n)
        self.calls += 1
        self.positions += n
        return np.zeros((n, POLICY_SIZE), dtype=np.float32), np.zeros(n, dtype=np.float32)


@pytest.fixture
def uniform():
    return UniformEvaluator()
