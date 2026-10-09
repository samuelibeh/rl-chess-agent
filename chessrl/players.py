from __future__ import annotations

from typing import Protocol

import chess
import chess.engine
import numpy as np

from .mcts import BatchedSearch, Evaluator, SearchConfig, Tree

_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
_MATE = 1000.0


class Player(Protocol):
    name: str

    def choose(self, board: chess.Board) -> chess.Move: ...

    def close(self) -> None: ...


class RandomPlayer:
    name = "random"

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()

    def choose(self, board: chess.Board) -> chess.Move:
        moves = list(board.legal_moves)
        return moves[int(self.rng.integers(len(moves)))]

    def close(self) -> None:
        pass


def material(board: chess.Board, color: chess.Color) -> float:
    return sum(
        _VALUES[p.piece_type] * (1 if p.color == color else -1)
        for p in board.piece_map().values()
    )


def _negamax(board: chess.Board, depth: int) -> float:
    """Material score for the side to move, with mate and draw detection."""
    if board.is_checkmate():
        return -_MATE - depth
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    if depth == 0:
        return material(board, board.turn)
    best = -float("inf")
    for move in board.legal_moves:
        board.push(move)
        best = max(best, -_negamax(board, depth - 1))
        board.pop()
    return best


class MaterialPlayer:
    """Fixed-depth material-only search. depth=1 looks one move ahead (greedy), depth=2 also
    considers the opponent's best reply. Both take mates in one. Ties are broken at random."""

    def __init__(self, depth: int = 1, rng: np.random.Generator | None = None):
        self.depth = depth
        self.name = f"material-d{depth}"
        self.rng = rng or np.random.default_rng()

    def choose(self, board: chess.Board) -> chess.Move:
        scored = []
        for move in board.legal_moves:
            board.push(move)
            scored.append((-_negamax(board, self.depth - 1), move))
            board.pop()
        best = max(s for s, _ in scored)
        top = [m for s, m in scored if s == best]
        return top[int(self.rng.integers(len(top)))]

    def close(self) -> None:
        pass


class MCTSPlayer:
    """Plays the most visited move of a PUCT search (no noise, temperature zero)."""

    def __init__(self, evaluator: Evaluator, sims: int = 64, leaves_per_round: int = 8,
                 c_puct: float = 1.5, rng: np.random.Generator | None = None, name: str | None = None):
        cfg = SearchConfig(sims=sims, c_puct=c_puct, leaves_per_round=leaves_per_round)
        self.search = BatchedSearch(evaluator, cfg, rng)
        self.name = name or f"mcts-{sims}"

    def choose(self, board: chess.Board) -> chess.Move:
        tree = Tree(board)
        self.search.search([tree], add_noise=False)
        return tree.best_move()

    def close(self) -> None:
        pass


class StockfishPlayer:
    """Stockfish through UCI, strength-limited to `elo` (Stockfish accepts roughly 1320 and up)."""

    def __init__(self, path: str, elo: int = 1350, movetime: float = 0.05):
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
        self.limit = chess.engine.Limit(time=movetime)
        self.name = f"stockfish-elo{elo}"

    def choose(self, board: chess.Board) -> chess.Move:
        return self.engine.play(board, self.limit).move

    def close(self) -> None:
        self.engine.quit()
