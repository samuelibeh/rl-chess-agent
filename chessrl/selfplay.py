"""Self-play with continuous batching: a fixed number of games are kept in flight, every move
all of them search together so one network call serves many games, and a finished game is
replaced by a new one immediately so batches stay full until the last games drain."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import chess
import numpy as np

from .encoding import encode_board
from .mcts import BatchedSearch, Evaluator, SearchConfig, Tree
from .replay import Sample


@dataclass
class SelfPlayConfig:
    games: int = 16
    concurrent: int = 8
    temp_moves: int = 30      # sample moves in proportion to visits for this many plies, then play the most visited
    max_plies: int = 200      # games still running at this length are scored as draws
    search: SearchConfig = field(default_factory=SearchConfig)


@dataclass
class SelfPlayStats:
    games: int = 0
    white_wins: int = 0
    black_wins: int = 0
    draws: int = 0
    max_ply_draws: int = 0
    positions: int = 0
    nn_calls: int = 0
    nn_positions: int = 0
    seconds: float = 0.0

    @property
    def avg_batch(self) -> float:
        return self.nn_positions / self.nn_calls if self.nn_calls else 0.0

    @property
    def positions_per_sec(self) -> float:
        return self.positions / self.seconds if self.seconds else 0.0

    @property
    def mean_game_length(self) -> float:
        return self.positions / self.games if self.games else 0.0


class _Game:
    def __init__(self):
        self.board = chess.Board()
        self.history: list[tuple[np.ndarray, np.ndarray, np.ndarray, bool]] = []


def _result(board: chess.Board) -> int | None:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0
    return 1 if outcome.winner == chess.WHITE else -1


def _finish(game: _Game, result: int) -> list[Sample]:
    return [
        Sample(planes.astype(np.float16), idx.astype(np.int16), pi.astype(np.float32),
               float(result if white_to_move else -result))
        for planes, idx, pi, white_to_move in game.history
    ]


def run_selfplay(evaluator: Evaluator, cfg: SelfPlayConfig,
                 rng: np.random.Generator | None = None) -> tuple[list[Sample], SelfPlayStats]:
    rng = rng or np.random.default_rng()
    search = BatchedSearch(evaluator, cfg.search, rng)
    stats = SelfPlayStats()
    samples: list[Sample] = []
    active: list[_Game] = []
    started = 0
    calls0 = getattr(evaluator, "calls", 0)
    positions0 = getattr(evaluator, "positions", 0)
    t0 = time.perf_counter()

    while started < cfg.games or active:
        while started < cfg.games and len(active) < cfg.concurrent:
            active.append(_Game())
            started += 1

        trees = [Tree(g.board) for g in active]
        search.search(trees, add_noise=True)

        still_active: list[_Game] = []
        for game, tree in zip(active, trees):
            counts = tree.visit_counts()
            pi = counts / counts.sum()
            ply = len(game.board.move_stack)
            game.history.append((
                encode_board(game.board), tree.root.move_idx.copy(), pi, game.board.turn == chess.WHITE,
            ))
            if ply < cfg.temp_moves:
                choice = int(rng.choice(len(pi), p=pi))
            else:
                choice = int(rng.choice(np.flatnonzero(counts == counts.max())))
            game.board.push(tree.root.moves[choice])
            stats.positions += 1

            result = _result(game.board)
            if result is None and len(game.board.move_stack) >= cfg.max_plies:
                result = 0
                stats.max_ply_draws += 1
            if result is None:
                still_active.append(game)
                continue
            samples.extend(_finish(game, result))
            stats.games += 1
            stats.white_wins += result == 1
            stats.black_wins += result == -1
            stats.draws += result == 0
        active = still_active

    stats.seconds = time.perf_counter() - t0
    stats.nn_calls = getattr(evaluator, "calls", 0) - calls0
    stats.nn_positions = getattr(evaluator, "positions", 0) - positions0
    return samples, stats
