import json

import numpy as np
import torch

from chessrl.network import PolicyValueNet, load_checkpoint
from chessrl.replay import ReplayBuffer, Sample
from chessrl.train import TrainConfig, train, train_step


def test_train_step_fits_a_fixed_batch():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    net = PolicyValueNet(channels=16, blocks=1)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    buf = ReplayBuffer(64)
    for _ in range(32):
        idx = np.sort(rng.choice(4672, size=5, replace=False)).astype(np.int16)
        pi = rng.dirichlet(np.ones(5)).astype(np.float32)
        buf.add_many([Sample(rng.random((19, 8, 8)).astype(np.float16), idx, pi, float(rng.choice([-1, 0, 1])))])
    batch = buf.sample_batch(32, np.random.default_rng(1))
    _, _, pi, _ = batch
    entropy = -(pi * torch.log(pi.clamp_min(1e-12))).sum(1).mean().item()
    first = train_step(net, opt, batch)
    for _ in range(80):
        last = train_step(net, opt, batch)
    # cross-entropy can only fall to the entropy of the targets, so compare the KL gap
    kl_first, kl_last = first["policy_loss"] - entropy, last["policy_loss"] - entropy
    assert kl_last < 0.5 * kl_first, (kl_first, kl_last)
    assert last["value_loss"] < first["value_loss"] * 0.5


def test_illegal_moves_are_masked_out_of_the_policy_loss():
    net = PolicyValueNet(channels=8, blocks=1)
    opt = torch.optim.SGD(net.parameters(), lr=0.0)
    buf = ReplayBuffer(4)
    buf.add_many([Sample(np.zeros((19, 8, 8), np.float16), np.array([5, 9], np.int16),
                         np.array([0.5, 0.5], np.float32), 0.0)])
    stats = train_step(net, opt, buf.sample_batch(1, np.random.default_rng(0)))
    # uniform over 2 legal moves at most: loss can never be as large as log(4672)
    assert stats["policy_loss"] < 3.0


def test_end_to_end_training_smoke(tmp_path):
    cfg = TrainConfig(iterations=2, games_per_iter=4, concurrent_games=4, sims=4, channels=8, blocks=1,
                      batch_size=16, steps_per_iter=3, min_buffer=16, max_plies=24, eval_every=2,
                      eval_games=2, eval_sims=4, device="cpu", out_dir=str(tmp_path / "run"))
    history = train(cfg, log=lambda _: None)
    assert len(history) == 2 and "loss" in history[-1] and "vs_random_score" in history[-1]
    rows = [json.loads(l) for l in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert [r["iteration"] for r in rows] == [1, 2]
    model = load_checkpoint(tmp_path / "run" / "latest.pt")
    assert model.config() == {"channels": 8, "blocks": 1}
