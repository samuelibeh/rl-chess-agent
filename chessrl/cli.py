from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from .evaluate import play_match
from .evaluator import NetEvaluator
from .mcts import SearchConfig
from .network import PolicyValueNet, load_checkpoint, pick_device
from .players import MaterialPlayer, MCTSPlayer, RandomPlayer, StockfishPlayer
from .selfplay import SelfPlayConfig, run_selfplay
from .train import TrainConfig, train


def _train(a) -> None:
    cfg = TrainConfig(
        iterations=a.iterations, games_per_iter=a.games_per_iter, concurrent_games=a.concurrent_games,
        sims=a.sims, leaves_per_round=a.leaves_per_round, channels=a.channels, blocks=a.blocks,
        batch_size=a.batch_size, steps_per_iter=a.steps_per_iter, lr=a.lr, max_plies=a.max_plies,
        eval_every=a.eval_every, eval_games=a.eval_games, eval_sims=a.eval_sims,
        device=a.device, amp=a.amp, seed=a.seed, out_dir=a.out_dir,
    )
    train(cfg)


def _eval(a) -> None:
    device = pick_device(a.device)
    rng = np.random.default_rng(a.seed)
    model = load_checkpoint(a.checkpoint, device)
    me = MCTSPlayer(NetEvaluator(model, device, amp=a.amp), sims=a.sims, leaves_per_round=a.leaves_per_round, rng=rng)
    if a.opponent == "random":
        opp = RandomPlayer(rng)
    elif a.opponent in ("material1", "material2"):
        opp = MaterialPlayer(int(a.opponent[-1]), rng)
    elif a.opponent == "stockfish":
        opp = StockfishPlayer(a.stockfish_path, a.stockfish_elo)
    else:
        other = load_checkpoint(a.opponent, device)
        opp = MCTSPlayer(NetEvaluator(other, device), sims=a.sims, leaves_per_round=a.leaves_per_round, rng=rng)
    try:
        res = play_match(me, opp, a.games, rng, max_plies=a.max_plies)
    finally:
        opp.close()
    print(f"{me.name} vs {opp.name}: {res.summary()}")


def _bench(a) -> None:
    device = pick_device(a.device)
    torch.manual_seed(0)
    model = PolicyValueNet(a.channels, a.blocks).to(device).eval()
    ev = NetEvaluator(model, device, amp=a.amp)
    print(f"device: {device}  network: {a.channels} channels x {a.blocks} blocks  "
          f"params: {sum(p.numel() for p in model.parameters()):,}  amp: {ev.amp}")

    print("\nnetwork forward throughput (random positions):")
    for bs in (1, 8, 32, 128, 512):
        x = np.random.default_rng(0).random((bs, 19, 8, 8), dtype=np.float32)
        ev(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        reps = max(3, 256 // bs)
        t0 = time.perf_counter()
        for _ in range(reps):
            ev(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"  batch {bs:>4}: {bs * reps / dt:>9.0f} positions/s  ({dt / reps * 1000:.1f} ms per call)")

    print(f"\nself-play throughput ({a.sims} sims per move, {a.leaves_per_round} leaf per tree per round):")
    for conc in a.concurrent:
        cfg = SelfPlayConfig(games=conc * 2, concurrent=conc, max_plies=a.max_plies,
                             search=SearchConfig(sims=a.sims, leaves_per_round=a.leaves_per_round))
        ev.reset_stats()
        _, st = run_selfplay(ev, cfg, np.random.default_rng(1))
        sims_per_s = st.positions * a.sims / st.seconds
        print(f"  {conc:>3} concurrent games: {st.positions_per_sec:>7.1f} moves/s, {sims_per_s:>8.0f} sims/s, "
              f"avg network batch {st.avg_batch:>5.1f}, {st.nn_calls} calls")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="chessrl", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="self-play training from random initialisation")
    t.add_argument("--iterations", type=int, default=20)
    t.add_argument("--games-per-iter", type=int, default=16)
    t.add_argument("--concurrent-games", type=int, default=16)
    t.add_argument("--sims", type=int, default=48)
    t.add_argument("--leaves-per-round", type=int, default=1)
    t.add_argument("--channels", type=int, default=64)
    t.add_argument("--blocks", type=int, default=4)
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--steps-per-iter", type=int, default=60)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--max-plies", type=int, default=200)
    t.add_argument("--eval-every", type=int, default=5)
    t.add_argument("--eval-games", type=int, default=10)
    t.add_argument("--eval-sims", type=int, default=32)
    t.add_argument("--device", default="auto")
    t.add_argument("--amp", action="store_true", help="bfloat16 autocast on CUDA")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out-dir", default="runs/default")
    t.set_defaults(fn=_train)

    e = sub.add_parser("eval", help="match a checkpoint against a baseline or another checkpoint")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--opponent", default="random",
                   help="random | material1 | material2 | stockfish | path to another checkpoint")
    e.add_argument("--games", type=int, default=20)
    e.add_argument("--sims", type=int, default=64)
    e.add_argument("--leaves-per-round", type=int, default=8)
    e.add_argument("--max-plies", type=int, default=300)
    e.add_argument("--stockfish-path", default="stockfish")
    e.add_argument("--stockfish-elo", type=int, default=1350)
    e.add_argument("--device", default="auto")
    e.add_argument("--amp", action="store_true")
    e.add_argument("--seed", type=int, default=0)
    e.set_defaults(fn=_eval)

    b = sub.add_parser("bench", help="measure network and self-play throughput on this machine")
    b.add_argument("--device", default="auto")
    b.add_argument("--channels", type=int, default=64)
    b.add_argument("--blocks", type=int, default=4)
    b.add_argument("--sims", type=int, default=32)
    b.add_argument("--leaves-per-round", type=int, default=1)
    b.add_argument("--concurrent", type=int, nargs="+", default=[1, 8, 32])
    b.add_argument("--max-plies", type=int, default=40)
    b.add_argument("--amp", action="store_true")
    b.set_defaults(fn=_bench)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
