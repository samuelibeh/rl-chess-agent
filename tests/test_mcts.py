import chess
import numpy as np

from chessrl.mcts import BatchedSearch, SearchConfig, Tree, make_node

MATE_IN_ONE = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"   # Ra8#


def search(ev, fen_or_boards, sims=64, lpr=1, noise=False, seed=0):
    boards = [chess.Board(f) if isinstance(f, str) else f for f in fen_or_boards]
    trees = [Tree(b) for b in boards]
    BatchedSearch(ev, SearchConfig(sims=sims, leaves_per_round=lpr), np.random.default_rng(seed)).search(trees, add_noise=noise)
    return trees


def test_finds_mate_in_one_with_an_uninformed_network(uniform):
    (tree,) = search(uniform, [MATE_IN_ONE], sims=60)
    assert tree.best_move().uci() == "a1a8"


def test_visit_counts_sum_to_sims_and_board_is_restored(uniform):
    board = chess.Board()
    board.push_san("e4")
    fen, stack = board.fen(), list(board.move_stack)
    (tree,) = search(uniform, [board], sims=50)
    assert tree.visit_counts().sum() == 50
    assert board.fen() == fen and list(board.move_stack) == stack


def test_results_do_not_depend_on_what_else_is_in_the_batch(uniform):
    fens = [MATE_IN_ONE, chess.STARTING_FEN, "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"]
    (alone,) = search(uniform, [fens[2]], sims=40)
    together = search(uniform, fens, sims=40)[2]
    assert np.array_equal(alone.visit_counts(), together.visit_counts())


def test_network_calls_batch_all_trees(uniform):
    search(uniform, [chess.STARTING_FEN] * 5, sims=6)
    assert uniform.batch_sizes[0] == 5, "all roots expanded in one call"
    assert max(uniform.batch_sizes) == 5 and uniform.batch_sizes.count(5) >= 5


def check_tree(node, depth=0):
    """No virtual loss may remain: every edge's total value is bounded by its visit count."""
    if not node.expanded:
        return 0
    assert (node.edge_n >= 0).all()
    assert (np.abs(node.edge_w) <= node.edge_n + 1e-9).all()
    n = 1
    for child in node.children:
        if child is not None:
            n += check_tree(child, depth + 1)
    return n


def test_virtual_loss_leaves_no_residue_and_total_visits_are_exact(uniform):
    for lpr in (2, 8, 16):
        (tree,) = search(uniform, [chess.STARTING_FEN], sims=100, lpr=lpr)
        assert tree.visit_counts().sum() == 100
        check_tree(tree.root)
        assert max(uniform.batch_sizes) <= lpr


def test_multiple_leaves_per_round_enlarge_batches(uniform):
    search(uniform, [chess.STARTING_FEN] * 2, sims=64, lpr=8)
    assert max(uniform.batch_sizes) > 2


def test_terminal_positions():
    assert make_node(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")).terminal_value == 0.0      # stalemate
    assert make_node(chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")).terminal_value == 0.0      # K vs K
    assert make_node(chess.Board("R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1")).terminal_value == -1.0  # mated
    assert make_node(chess.Board()).terminal_value is None


def test_cannot_search_a_finished_game():
    import pytest
    with pytest.raises(ValueError):
        Tree(chess.Board("R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1"))


def test_dirichlet_noise_changes_root_priors_only_when_asked(uniform):
    (plain,) = search(uniform, [chess.STARTING_FEN], sims=1, noise=False)
    (noisy,) = search(uniform, [chess.STARTING_FEN], sims=1, noise=True)
    assert np.allclose(plain.root.priors, 1 / 20)
    assert not np.allclose(noisy.root.priors, 1 / 20)
    assert abs(noisy.root.priors.sum() - 1) < 1e-9


def test_prefers_winning_a_free_queen_with_a_material_value_function():
    from chessrl.encoding import POLICY_SIZE
    from chessrl.players import material

    def evaluator(planes):
        # value from the side to move's view: own minus opponent material, squashed
        w = np.array([1, 3, 3, 5, 9, 0], dtype=np.float32)
        own = (planes[:, :6].sum(axis=(2, 3)) * w).sum(1)
        opp = (planes[:, 6:12].sum(axis=(2, 3)) * w).sum(1)
        return np.zeros((len(planes), POLICY_SIZE), np.float32), np.tanh((own - opp) / 10).astype(np.float32)

    board = chess.Board("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1")   # Rxd5 wins the queen
    (tree,) = search(evaluator, [board], sims=120)
    assert tree.best_move().uci() == "d1d5"
    assert material(board, chess.WHITE) == -4
