"""Board and move encoding, always from the point of view of the side to move.

Positions are mirrored vertically when Black is to move, so the network only ever sees
"my pieces move up the board". Moves use the AlphaZero 8x8x73 layout: 56 queen-like moves
(8 directions x 7 distances), 8 knight moves and 9 underpromotions (3 pieces x 3 directions).
Queen promotions are queen-like moves. The policy index is `plane * 64 + from_square`, which
matches flattening a (73, 8, 8) network output.
"""

from __future__ import annotations

import chess
import numpy as np

NUM_INPUT_PLANES = 19
NUM_MOVE_PLANES = 73
POLICY_SIZE = NUM_MOVE_PLANES * 64

_PIECE_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
_QUEEN_DIRS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
_KNIGHT_OFFSETS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
_DIR_PLANE = {d: i for i, d in enumerate(_QUEEN_DIRS)}
_KNIGHT_PLANE = {d: i for i, d in enumerate(_KNIGHT_OFFSETS)}
_UNDERPROMOTION = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}


def _bits(mask: int, flip: bool) -> np.ndarray:
    if flip:
        mask = chess.flip_vertical(mask)
    raw = np.frombuffer(mask.to_bytes(8, "little"), dtype=np.uint8)
    return np.unpackbits(raw, bitorder="little")


def encode_board(board: chess.Board) -> np.ndarray:
    """Returns a (19, 8, 8) float32 array: 6 own piece planes, 6 opponent piece planes,
    4 castling-right planes, en passant, halfmove clock, and a constant plane."""
    us = board.turn
    flip = us == chess.BLACK
    planes = np.zeros((NUM_INPUT_PLANES, 64), dtype=np.float32)
    for i, pt in enumerate(_PIECE_TYPES):
        planes[i] = _bits(board.pieces_mask(pt, us), flip)
        planes[6 + i] = _bits(board.pieces_mask(pt, not us), flip)
    planes[12] = board.has_kingside_castling_rights(us)
    planes[13] = board.has_queenside_castling_rights(us)
    planes[14] = board.has_kingside_castling_rights(not us)
    planes[15] = board.has_queenside_castling_rights(not us)
    if board.ep_square is not None:
        planes[16, board.ep_square ^ 56 if flip else board.ep_square] = 1.0
    planes[17] = min(board.halfmove_clock, 100) / 100.0
    planes[18] = 1.0
    return planes.reshape(NUM_INPUT_PLANES, 8, 8)


def move_to_index(move: chess.Move, flip: bool) -> int:
    """Policy index of `move`; pass flip=True when Black is the side to move."""
    f, t = move.from_square, move.to_square
    if flip:
        f ^= 56
        t ^= 56
    dx = (t & 7) - (f & 7)
    dy = (t >> 3) - (f >> 3)
    if move.promotion is not None and move.promotion != chess.QUEEN:
        plane = 64 + _UNDERPROMOTION[move.promotion] * 3 + (dx + 1)
    elif (dx, dy) in _KNIGHT_PLANE:
        plane = 56 + _KNIGHT_PLANE[(dx, dy)]
    else:
        sx = (dx > 0) - (dx < 0)
        sy = (dy > 0) - (dy < 0)
        plane = _DIR_PLANE[(sx, sy)] * 7 + max(abs(dx), abs(dy)) - 1
    return plane * 64 + f


def legal_moves_with_indices(board: chess.Board) -> tuple[list[chess.Move], np.ndarray]:
    flip = board.turn == chess.BLACK
    moves = list(board.legal_moves)
    idx = np.fromiter((move_to_index(m, flip) for m in moves), dtype=np.int32, count=len(moves))
    return moves, idx
