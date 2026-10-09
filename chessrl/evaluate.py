"""Strength measurement: matches against fixed opponents with colours alternated, shared random
openings for each colour pair, and an Elo difference with a confidence interval."""

from __future__ import annotations

import math
from dataclasses import dataclass

import chess
import numpy as np

from .players import Player


@dataclass
class MatchResult:
    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.5

    def elo_diff(self, z: float = 1.96) -> tuple[float, float, float]:
        """Elo of player A relative to B with a normal-approximation interval (about 95% for z=1.96)."""
        n = self.games
        if n == 0:
            return 0.0, -math.inf, math.inf
        s = self.score
        var = (self.wins * (1 - s) ** 2 + self.draws * (0.5 - s) ** 2 + self.losses * s ** 2) / n
        se = math.sqrt(var / n)
        return elo_from_score(s), elo_from_score(s - z * se), elo_from_score(s + z * se)

    def summary(self) -> str:
        elo, lo, hi = self.elo_diff()
        return (f"W{self.wins} D{self.draws} L{self.losses} over {self.games} games, "
                f"score {self.score:.3f}, Elo {elo:+.0f} [{lo:+.0f}, {hi:+.0f}]")


def elo_from_score(score: float) -> float:
    s = min(max(score, 1e-3), 1 - 1e-3)
    return -400.0 * math.log10(1.0 / s - 1.0)


def random_opening(plies: int, rng: np.random.Generator) -> list[chess.Move]:
    board = chess.Board()
    moves: list[chess.Move] = []
    for _ in range(plies):
        legal = list(board.legal_moves)
        if not legal:
            break
        move = legal[int(rng.integers(len(legal)))]
        board.push(move)
        moves.append(move)
        if board.is_game_over():
            return random_opening(plies, rng)
    return moves


def play_game(white: Player, black: Player, opening: list[chess.Move], max_plies: int = 300) -> int:
    """Returns 1 if White wins, -1 if Black wins, 0 for a draw (including the ply cap)."""
    board = chess.Board()
    for m in opening:
        board.push(m)
    while len(board.move_stack) < max_plies:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            return 0 if outcome.winner is None else (1 if outcome.winner == chess.WHITE else -1)
        board.push((white if board.turn == chess.WHITE else black).choose(board))
    return 0


def play_match(a: Player, b: Player, games: int, rng: np.random.Generator | None = None,
               opening_plies: int = 4, max_plies: int = 300) -> MatchResult:
    """Player A against player B. Games come in colour-swapped pairs that share an opening."""
    rng = rng or np.random.default_rng()
    result = MatchResult()
    for k in range((games + 1) // 2):
        opening = random_opening(opening_plies, rng)
        for a_is_white in (True, False):
            if result.games >= games:
                break
            white, black = (a, b) if a_is_white else (b, a)
            r = play_game(white, black, opening, max_plies)
            r = r if a_is_white else -r
            result.wins += r == 1
            result.draws += r == 0
            result.losses += r == -1
    return result
