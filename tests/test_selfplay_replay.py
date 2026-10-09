import chess
import numpy as np

from chessrl.mcts import SearchConfig
from chessrl.replay import ReplayBuffer, Sample
from chessrl.selfplay import SelfPlayConfig, _Game, _finish, _result, run_selfplay


def cfg(games, concurrent, sims=6, max_plies=30):
    return SelfPlayConfig(games=games, concurrent=concurrent, max_plies=max_plies, temp_moves=10,
                          search=SearchConfig(sims=sims))


def test_result_detection():
    assert _result(chess.Board()) is None
    assert _result(chess.Board("R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1")) == 1       # Black is mated
    assert _result(chess.Board("6k1/5ppp/8/8/8/8/5PPP/r5K1 w - - 0 1")) == -1      # White is mated
    assert _result(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")) == 0             # stalemate


def test_outcome_is_assigned_from_each_side_to_moves_point_of_view():
    g = _Game()
    plane = np.zeros((19, 8, 8), np.float32)
    idx, pi = np.array([1, 2]), np.array([0.5, 0.5])
    g.history = [(plane, idx, pi, True), (plane, idx, pi, False), (plane, idx, pi, True)]
    assert [s.z for s in _finish(g, 1)] == [1.0, -1.0, 1.0]
    assert [s.z for s in _finish(g, -1)] == [-1.0, 1.0, -1.0]
    assert [s.z for s in _finish(g, 0)] == [0.0, 0.0, 0.0]


def test_selfplay_produces_consistent_training_data(uniform):
    samples, stats = run_selfplay(uniform, cfg(games=5, concurrent=3), np.random.default_rng(0))
    assert stats.games == 5 and stats.white_wins + stats.black_wins + stats.draws == 5
    assert len(samples) == stats.positions
    for s in samples:
        assert s.planes.shape == (19, 8, 8) and s.planes.dtype == np.float16
        assert len(s.legal_idx) == len(s.pi) > 0
        assert abs(float(s.pi.sum()) - 1.0) < 1e-5
        assert s.z in (-1.0, 0.0, 1.0)
    assert stats.max_ply_draws <= stats.draws


def test_continuous_batching_keeps_batches_full_and_bounded(uniform):
    run_selfplay(uniform, cfg(games=6, concurrent=3, sims=4), np.random.default_rng(1))
    assert max(uniform.batch_sizes) <= 3
    assert uniform.batch_sizes[0] == 3, "first call evaluates the roots of all concurrent games together"
    stats_batch = sum(uniform.batch_sizes) / len(uniform.batch_sizes)
    assert stats_batch > 2.0


def test_concurrency_reduces_network_calls_for_the_same_work():
    from tests.conftest import UniformEvaluator
    a, b = UniformEvaluator(), UniformEvaluator()
    _, s1 = run_selfplay(a, cfg(games=4, concurrent=1, sims=4), np.random.default_rng(2))
    _, s4 = run_selfplay(b, cfg(games=4, concurrent=4, sims=4), np.random.default_rng(2))
    assert s4.nn_calls < s1.nn_calls / 2
    assert s4.avg_batch > s1.avg_batch


def test_stats_use_deltas_when_the_evaluator_is_reused(uniform):
    run_selfplay(uniform, cfg(games=2, concurrent=2, sims=3), np.random.default_rng(0))
    _, second = run_selfplay(uniform, cfg(games=2, concurrent=2, sims=3), np.random.default_rng(0))
    assert second.nn_positions < uniform.positions


def make_sample(i):
    return Sample(np.full((19, 8, 8), i, np.float16), np.array([i, i + 1], np.int16),
                  np.array([0.25, 0.75], np.float32), 1.0)


def test_replay_buffer_is_fifo_when_full():
    buf = ReplayBuffer(capacity=3)
    buf.add_many([make_sample(i) for i in range(5)])
    assert len(buf) == 3
    assert sorted(int(s.planes[0, 0, 0]) for s in buf._items) == [2, 3, 4]


def test_replay_batch_shapes_and_masks():
    buf = ReplayBuffer(10)
    buf.add_many([make_sample(i) for i in range(4)])
    planes, mask, pi, z = buf.sample_batch(6, np.random.default_rng(0))
    assert planes.shape == (6, 19, 8, 8) and mask.shape == pi.shape == (6, 4672) and z.shape == (6,)
    assert mask.sum(1).tolist() == [2] * 6
    assert (pi[~mask] == 0).all()
    assert np.allclose(pi.sum(1).numpy(), 1.0)
