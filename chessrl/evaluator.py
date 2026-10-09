from __future__ import annotations

import numpy as np
import torch

from .network import PolicyValueNet


class NetEvaluator:
    """Runs the network on batches of encoded positions and counts how well batches fill."""

    def __init__(self, model: PolicyValueNet, device: torch.device | str = "cpu",
                 amp: bool = False, max_batch: int = 1024):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.amp = amp and self.device.type == "cuda"
        self.max_batch = max_batch
        self.calls = 0
        self.positions = 0

    def reset_stats(self) -> None:
        self.calls = 0
        self.positions = 0

    @torch.inference_mode()
    def __call__(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits, values = [], []
        for start in range(0, len(planes), self.max_batch):
            x = torch.from_numpy(np.ascontiguousarray(planes[start:start + self.max_batch])).to(self.device)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
                lg, v = self.model(x)
            logits.append(lg.float().cpu().numpy())
            values.append(v.float().cpu().numpy())
            self.calls += 1
        self.positions += len(planes)
        return np.concatenate(logits), np.concatenate(values)
