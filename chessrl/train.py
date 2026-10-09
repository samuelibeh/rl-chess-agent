from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .evaluate import play_match
from .evaluator import NetEvaluator
from .mcts import SearchConfig
from .network import PolicyValueNet, pick_device, save_checkpoint
from .players import MCTSPlayer, RandomPlayer
from .replay import ReplayBuffer
from .selfplay import SelfPlayConfig, run_selfplay


@dataclass
class TrainConfig:
    iterations: int = 20
    games_per_iter: int = 16
    concurrent_games: int = 16
    sims: int = 48
    leaves_per_round: int = 1
    temp_moves: int = 30
    max_plies: int = 200
    channels: int = 64
    blocks: int = 4
    batch_size: int = 128
    steps_per_iter: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    buffer_size: int = 50_000
    min_buffer: int = 512
    eval_every: int = 5
    eval_games: int = 10
    eval_sims: int = 32
    device: str = "auto"
    amp: bool = False
    seed: int = 0
    out_dir: str = "runs/default"
    extra: dict = field(default_factory=dict)


def train_step(model: PolicyValueNet, opt: torch.optim.Optimizer, batch, amp: bool = False) -> dict:
    planes, mask, pi, z = batch
    model.train()
    device_type = planes.device.type
    with torch.autocast(device_type, dtype=torch.bfloat16, enabled=amp and device_type == "cuda"):
        logits, value = model(planes)
    logp = F.log_softmax(logits.float().masked_fill(~mask, -1e9), dim=1)
    policy_loss = -(pi * logp).sum(dim=1).mean()
    value_loss = F.mse_loss(value.float(), z)
    loss = policy_loss + value_loss
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return {"loss": loss.item(), "policy_loss": policy_loss.item(), "value_loss": value_loss.item()}


def train(cfg: TrainConfig, log=print) -> list[dict]:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = pick_device(cfg.device)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    model = PolicyValueNet(cfg.channels, cfg.blocks).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    buffer = ReplayBuffer(cfg.buffer_size)
    evaluator = NetEvaluator(model, device, amp=cfg.amp)
    sp_cfg = SelfPlayConfig(
        games=cfg.games_per_iter, concurrent=cfg.concurrent_games, temp_moves=cfg.temp_moves,
        max_plies=cfg.max_plies,
        search=SearchConfig(sims=cfg.sims, leaves_per_round=cfg.leaves_per_round),
    )

    history: list[dict] = []
    t_start = time.perf_counter()
    for it in range(1, cfg.iterations + 1):
        model.eval()
        samples, sp = run_selfplay(evaluator, sp_cfg, rng)
        buffer.add_many(samples)

        losses = []
        if len(buffer) >= cfg.min_buffer:
            for _ in range(cfg.steps_per_iter):
                batch = buffer.sample_batch(cfg.batch_size, rng, device)
                losses.append(train_step(model, opt, batch, cfg.amp))

        row = {
            "iteration": it,
            "elapsed_s": round(time.perf_counter() - t_start, 1),
            "games": sp.games, "white_wins": sp.white_wins, "black_wins": sp.black_wins,
            "draws": sp.draws, "max_ply_draws": sp.max_ply_draws,
            "mean_game_length": round(sp.mean_game_length, 1),
            "selfplay_positions_per_s": round(sp.positions_per_sec, 1),
            "avg_nn_batch": round(sp.avg_batch, 1),
            "buffer": len(buffer),
        }
        if losses:
            for k in ("loss", "policy_loss", "value_loss"):
                row[k] = round(float(np.mean([l[k] for l in losses])), 4)

        if cfg.eval_every and (it % cfg.eval_every == 0 or it == cfg.iterations):
            model.eval()
            player = MCTSPlayer(evaluator, sims=cfg.eval_sims, rng=rng)
            res = play_match(player, RandomPlayer(rng), cfg.eval_games, rng, max_plies=200)
            row["vs_random_score"] = round(res.score, 3)
            row["vs_random"] = res.summary()

        history.append(row)
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        save_checkpoint(model, out / "latest.pt", iteration=it)
        log(json.dumps(row))
    return history
