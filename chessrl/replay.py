from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .encoding import POLICY_SIZE


@dataclass
class Sample:
    planes: np.ndarray      # (19, 8, 8) float16
    legal_idx: np.ndarray   # (L,) int16 policy indices of the legal moves
    pi: np.ndarray          # (L,) float32 MCTS visit distribution over those moves
    z: float                # game outcome from the side to move's point of view


class ReplayBuffer:
    """Fixed-size FIFO of self-play positions."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._items: list[Sample] = []
        self._next = 0

    def __len__(self) -> int:
        return len(self._items)

    def add_many(self, samples: list[Sample]) -> None:
        for s in samples:
            if len(self._items) < self.capacity:
                self._items.append(s)
            else:
                self._items[self._next] = s
                self._next = (self._next + 1) % self.capacity

    def sample_batch(self, batch_size: int, rng: np.random.Generator, device: torch.device | str = "cpu"):
        picks = rng.integers(0, len(self._items), size=batch_size)
        planes = np.stack([self._items[i].planes for i in picks]).astype(np.float32)
        mask = np.zeros((batch_size, POLICY_SIZE), dtype=bool)
        pi = np.zeros((batch_size, POLICY_SIZE), dtype=np.float32)
        z = np.empty(batch_size, dtype=np.float32)
        for row, i in enumerate(picks):
            s = self._items[i]
            mask[row, s.legal_idx] = True
            pi[row, s.legal_idx] = s.pi
            z[row] = s.z
        return (
            torch.from_numpy(planes).to(device),
            torch.from_numpy(mask).to(device),
            torch.from_numpy(pi).to(device),
            torch.from_numpy(z).to(device),
        )
