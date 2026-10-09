import chess
import numpy as np

from chessrl.encoding import (NUM_INPUT_PLANES, POLICY_SIZE, encode_board,
                              legal_moves_with_indices, move_to_index)


def random_positions(n, seed=0):
    rng = np.random.default_rng(seed)
    board = chess.Board()
    out = []
    while len(out) < n:
        if board.is_game_over():
            board = chess.Board()
        moves = list(board.legal_moves)
        board.push(moves[int(rng.integers(len(moves)))])
        out.append(board.copy())
    return out


def test_start_position_planes():
    p = encode_board(chess.Board())
    assert p.shape == (NUM_INPUT_PLANES, 8, 8)
    assert p[0].sum() == 8 and p[0, 1].sum() == 8          # own pawns on rank 2
    assert p[6, 6].sum() == 8                              # opponent pawns on rank 7
    assert p[5, 0, 4] == 1 and p[11, 7, 4] == 1            # kings
    assert all(p[i].min() == 1 for i in (12, 13, 14, 15))  # all castling rights
    assert p[18].min() == 1 and p[16].sum() == 0


def test_black_to_move_is_seen_from_its_own_side():
    board = chess.Board()
    board.push_san("e4")
    p = encode_board(board)
    assert p[0, 1].sum() == 8, "Black's pawns should appear on the second row after mirroring"
    assert p[6, 4, 4] == 1, "White's e4 pawn is an opponent pawn; e4 (row 3) mirrors to row 4"
    assert p[16, 5, 4] == 1, "en passant square e3 (row 2) mirrors to row 5"


def test_mirrored_positions_encode_identically():
    for board in random_positions(300):
        mirror = board.mirror()
        assert np.array_equal(encode_board(board), encode_board(mirror))
        moves, idx = legal_moves_with_indices(board)
        mmoves, midx = legal_moves_with_indices(mirror)
        by_move = {(m.from_square ^ 56, m.to_square ^ 56, m.promotion): i for m, i in zip(moves, idx)}
        for m, i in zip(mmoves, midx):
            assert by_move[(m.from_square, m.to_square, m.promotion)] == i


def test_every_legal_move_gets_a_distinct_in_range_index():
    fens = [
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",                   # castling
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        "1n2k3/P1P1P3/8/8/8/8/p1p1p3/1N2K3 w - - 0 1",            # promotions and capture-promotions
        "1n2k3/P1P1P3/8/8/8/8/p1p1p3/1N2K3 b - - 0 1",
        "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",  # en passant
        "4k3/8/8/8/8/8/8/Q3K2Q w - - 0 1",                        # many queen moves
    ]
    boards = [chess.Board(f) for f in fens] + random_positions(1500, seed=3)
    for board in boards:
        moves, idx = legal_moves_with_indices(board)
        if not moves:
            continue
        assert len(set(idx.tolist())) == len(moves), board.fen()
        assert idx.min() >= 0 and idx.max() < POLICY_SIZE


def test_known_move_indices():
    b = chess.Board()
    assert move_to_index(chess.Move.from_uci("e2e4"), False) == 1 * 64 + 12      # north, distance 2
    assert move_to_index(chess.Move.from_uci("g1f3"), False) == 63 * 64 + 6      # knight offset (-1, +2)
    promo = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    got = {m.promotion: i for m, i in zip(*legal_moves_with_indices(promo)) if m.from_square == chess.A7}
    assert got[chess.QUEEN] == 0 * 64 + 48                                       # queen promotion is a plain step
    assert got[chess.KNIGHT] == (64 + 0 * 3 + 1) * 64 + 48
    assert got[chess.BISHOP] == (64 + 1 * 3 + 1) * 64 + 48
    assert got[chess.ROOK] == (64 + 2 * 3 + 1) * 64 + 48
    assert b.is_valid()


def test_castling_is_a_two_square_king_move():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    moves, idx = legal_moves_with_indices(board)
    table = {m.uci(): int(i) for m, i in zip(moves, idx)}
    assert table["e1g1"] == (2 * 7 + 1) * 64 + 4      # east, distance 2, from e1
    assert table["e1c1"] == (6 * 7 + 1) * 64 + 4      # west, distance 2


def test_halfmove_clock_plane():
    board = chess.Board("4k3/8/8/8/8/8/8/4K2R w - - 50 80")
    assert np.allclose(encode_board(board)[17], 0.5)
