"""Batched Monte Carlo tree search (PUCT) over many independent games.

One call to `BatchedSearch.search` advances every tree in lockstep: each round every tree
selects up to `leaves_per_round` leaves, all non-terminal leaves are evaluated by the network
in a single batch, then the results are backed up. Batch size per network call is therefore
(number of trees) x (leaves per round), which is what keeps a GPU busy.

Edge statistics are stored from the point of view of the player to move at the node that owns
the edge, so selection is a plain argmax of Q + U and a backup flips the sign once per ply.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import chess
import numpy as np

from .encoding import encode_board, legal_moves_with_indices

_NO_INDEX = np.zeros(0, dtype=np.int32)


class Evaluator(Protocol):
    def __call__(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """planes: (B, 19, 8, 8) -> (policy logits (B, 4672), values (B,) in [-1, 1])."""


@dataclass
class SearchConfig:
    sims: int = 64
    c_puct: float = 1.5
    leaves_per_round: int = 1
    virtual_loss: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25


class Node:
    __slots__ = (
        "moves", "move_idx", "terminal_value", "priors", "edge_n", "edge_w",
        "children", "expanded", "inflight",
    )

    def __init__(self, moves: list[chess.Move], move_idx: np.ndarray, terminal_value: float | None):
        self.moves = moves
        self.move_idx = move_idx
        self.terminal_value = terminal_value  # from the side to move's point of view
        self.priors: np.ndarray | None = None
        self.edge_n: np.ndarray | None = None
        self.edge_w: np.ndarray | None = None
        self.children: list[Node | None] = []
        self.expanded = False
        self.inflight = False

    def expand(self, logits: np.ndarray) -> None:
        sel = logits[self.move_idx].astype(np.float64)
        sel -= sel.max()
        p = np.exp(sel)
        self.priors = p / p.sum()
        self.edge_n = np.zeros(len(self.moves))
        self.edge_w = np.zeros(len(self.moves))
        self.children = [None] * len(self.moves)
        self.expanded = True
        self.inflight = False


def make_node(board: chess.Board) -> Node:
    moves, idx = legal_moves_with_indices(board)
    if not moves:
        return Node([], _NO_INDEX, -1.0 if board.is_check() else 0.0)
    if board.is_insufficient_material() or board.halfmove_clock >= 100 or board.is_repetition(3):
        return Node([], _NO_INDEX, 0.0)
    return Node(moves, idx, None)


class Tree:
    """A search tree rooted at `board`. The board is pushed and popped during search and is
    always restored, so it can be the live game board (repetition history included)."""

    def __init__(self, board: chess.Board):
        self.board = board
        self.root = make_node(board)
        if self.root.terminal_value is not None:
            raise ValueError("cannot search a finished game")

    def visit_counts(self) -> np.ndarray:
        return self.root.edge_n.copy()

    def best_move(self) -> chess.Move:
        return self.root.moves[int(np.argmax(self.root.edge_n))]


class BatchedSearch:
    def __init__(self, evaluator: Evaluator, cfg: SearchConfig, rng: np.random.Generator | None = None):
        self.evaluator = evaluator
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()

    def search(self, trees: list[Tree], sims: int | None = None, add_noise: bool = False) -> None:
        sims = self.cfg.sims if sims is None else sims
        self._expand_roots(trees)
        if add_noise:
            for t in trees:
                self._add_noise(t.root)

        done = [0] * len(trees)
        while any(d < sims for d in done):
            pending: list[tuple[int, list, Node, np.ndarray]] = []
            for ti, tree in enumerate(trees):
                for _ in range(min(self.cfg.leaves_per_round, sims - done[ti])):
                    kind, path, node, payload = self._select(tree)
                    if kind == "blocked":
                        break
                    if kind == "terminal":
                        self._backup(path, node.terminal_value)
                        done[ti] += 1
                    else:
                        pending.append((ti, path, node, payload))
            if pending:
                logits, values = self.evaluator(np.stack([p[3] for p in pending]))
                for b, (ti, path, node, _) in enumerate(pending):
                    node.expand(logits[b])
                    self._backup(path, float(values[b]))
                    done[ti] += 1

    def _expand_roots(self, trees: list[Tree]) -> None:
        todo = [t for t in trees if not t.root.expanded]
        if not todo:
            return
        logits, _ = self.evaluator(np.stack([encode_board(t.board) for t in todo]))
        for b, t in enumerate(todo):
            t.root.expand(logits[b])

    def _add_noise(self, root: Node) -> None:
        noise = self.rng.dirichlet([self.cfg.dirichlet_alpha] * len(root.moves))
        eps = self.cfg.dirichlet_eps
        root.priors = (1 - eps) * root.priors + eps * noise

    def _pick(self, node: Node) -> int:
        n = node.edge_n
        q = np.divide(node.edge_w, n, out=np.zeros_like(node.edge_w), where=n > 0)
        u = self.cfg.c_puct * node.priors * math.sqrt(max(n.sum(), 1.0)) / (1.0 + n)
        return int(np.argmax(q + u))

    def _select(self, tree: Tree):
        board, node, path, pushed = tree.board, tree.root, [], 0
        vl = self.cfg.virtual_loss
        try:
            while True:
                if node.terminal_value is not None:
                    return "terminal", path, node, None
                if not node.expanded:
                    if node.inflight:
                        for n, i in path:
                            n.edge_n[i] -= 1
                            n.edge_w[i] += vl
                        return "blocked", path, node, None
                    node.inflight = True
                    return "eval", path, node, encode_board(board)
                i = self._pick(node)
                node.edge_n[i] += 1
                node.edge_w[i] -= vl
                path.append((node, i))
                board.push(node.moves[i])
                pushed += 1
                child = node.children[i]
                if child is None:
                    child = make_node(board)
                    node.children[i] = child
                node = child
        finally:
            for _ in range(pushed):
                board.pop()

    def _backup(self, path: list, leaf_value: float) -> None:
        v = leaf_value
        for node, i in reversed(path):
            v = -v
            node.edge_w[i] += v + self.cfg.virtual_loss
