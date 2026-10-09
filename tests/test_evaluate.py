import chess
import numpy as np

from chessrl.evaluate import MatchResult, elo_from_score, play_game, play_match, random_opening
from chessrl.players import MaterialPlayer, RandomPlayer


def test_elo_formula():
    assert elo_from_score(0.5) == 0
    assert abs(elo_from_score(0.75) - 190.8) < 0.1
    assert abs(elo_from_score(0.25) + elo_from_score(0.75)) < 1e-9
    assert np.isfinite(elo_from_score(1.0)) and np.isfinite(elo_from_score(0.0))


def test_match_result_interval_contains_the_estimate_and_narrows_with_games():
    small, large = MatchResult(6, 2, 2), MatchResult(60, 20, 20)
    assert small.score == large.score == 0.7
    e1, lo1, hi1 = small.elo_diff()
    e2, lo2, hi2 = large.elo_diff()
    assert lo1 < e1 < hi1 and lo2 < e2 < hi2
    assert (hi2 - lo2) < (hi1 - lo1)
    assert "W60 D20 L20" in large.summary()


class Recorder:
    def __init__(self, name):
        self.name, self.colours, self.rng = name, [], np.random.default_rng(0)

    def choose(self, board):
        self.colours.append(board.turn)
        return list(board.legal_moves)[0]

    def close(self):
        pass


def test_colours_alternate_and_game_count_is_exact():
    a, b = Recorder("a"), Recorder("b")
    result = play_match(a, b, games=5, rng=np.random.default_rng(0), opening_plies=2, max_plies=20)
    assert result.games == 5
    assert chess.WHITE in a.colours and chess.BLACK in a.colours


def test_ply_cap_scores_a_draw():
    a, b = Recorder("a"), Recorder("b")
    assert play_game(a, b, [], max_plies=6) == 0


def test_openings_are_legal_and_not_terminal():
    rng = np.random.default_rng(5)
    for _ in range(20):
        board = chess.Board()
        for m in random_opening(6, rng):
            assert m in board.legal_moves
            board.push(m)
        assert not board.is_game_over()


def test_material_players_take_free_material_and_mate():
    rng = np.random.default_rng(0)
    hanging_queen = chess.Board("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1")
    assert MaterialPlayer(1, rng).choose(hanging_queen).uci() == "d1d5"
    mate = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    assert MaterialPlayer(1, rng).choose(mate).uci() == "a1a8"
    assert MaterialPlayer(2, rng).choose(mate).uci() == "a1a8"


def test_depth_two_avoids_a_poisoned_capture_that_depth_one_takes():
    # Qxd5?? loses the queen to the pawn on e6... depth 1 sees a free pawn, depth 2 sees the recapture.
    board = chess.Board("4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1")
    rng = np.random.default_rng(0)
    assert MaterialPlayer(1, rng).choose(board).uci() == "d1d5"
    assert MaterialPlayer(2, rng).choose(board).uci() != "d1d5"


def test_random_player_only_plays_legal_moves():
    rng = np.random.default_rng(0)
    p, board = RandomPlayer(rng), chess.Board()
    for _ in range(40):
        if board.is_game_over():
            break
        m = p.choose(board)
        assert m in board.legal_moves
        board.push(m)
