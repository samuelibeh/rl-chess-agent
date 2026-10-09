from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .encoding import NUM_INPUT_PLANES, NUM_MOVE_PLANES, POLICY_SIZE


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.body(x))


class PolicyValueNet(nn.Module):
    """Residual tower with an AlphaZero-style policy head (73 move planes) and a tanh value head."""

    def __init__(self, channels: int = 64, blocks: int = 4):
        super().__init__()
        self.channels = channels
        self.blocks = blocks
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_INPUT_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, NUM_MOVE_PLANES, 1),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.tower(self.stem(x))
        logits = self.policy_head(h).flatten(1)
        assert logits.shape[1] == POLICY_SIZE
        return logits, self.value_head(h).squeeze(1)

    def config(self) -> dict:
        return {"channels": self.channels, "blocks": self.blocks}


def save_checkpoint(model: PolicyValueNet, path: str | Path, **extra) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": model.config(), **extra}, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> PolicyValueNet:
    blob = torch.load(path, map_location=device, weights_only=False)
    model = PolicyValueNet(**blob["config"])
    model.load_state_dict(blob["model"])
    return model.to(device).eval()


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
