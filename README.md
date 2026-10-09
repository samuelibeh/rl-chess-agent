# chessrl

An AlphaZero-style chess agent that learns only from games it plays against itself. There is no human game data, no opening book and no
hand-written evaluation function in the learning loop. PyTorch for the network, [python-chess](https://python-chess.readthedocs.io) for the rules.

```
        ┌────────────────────── replay buffer ◀────────────────────────┐
        │                                                               │ (position, MCTS visit
        ▼                                                               │  distribution, outcome)
  train policy/value net  ──weights──▶  batched self-play (MCTS + net) ─┘
        ▲                                        │
        └────────────── repeat ──────────────────┘            evaluate vs fixed opponents
```

## Algorithm

* **Network** (`network.py`): residual tower (default 64 channels, 4 blocks) over 19 input planes (own pieces, opponent pieces, castling
  rights, en passant, halfmove clock, constant plane), always encoded from the side to move's point of view. Policy head outputs the
  AlphaZero 8x8x73 move layout (4,672 logits); value head is a tanh scalar. See `encoding.py`.
* **Search** (`mcts.py`): PUCT Monte Carlo tree search. Priors come from the policy softmax over legal moves only, leaf values from the value
  head, with exact values for checkmate, stalemate, insufficient material, the 50-move rule and threefold repetition. Dirichlet noise at the root during self-play.
* **Training** (`train.py`): cross-entropy of the policy against the root visit distribution (illegal moves masked out) plus mean-squared
  error of the value against the final game result, from the side to move's point of view. AdamW.
* **Data**: positions from the most recent games only (FIFO buffer). Games longer than `--max-plies` are scored as draws.

## Batching self-play for the GPU

A single position is a tiny workload for a GPU, so the search is organised to hand the network large batches (`mcts.py`, `selfplay.py`):

1. **Many games at once.** `--concurrent-games` games are kept in flight. Each simulation round, every game's tree selects a leaf, and all those
   leaves go to the network in **one forward pass**. Batch size is therefore about the number of concurrent games.
2. **Continuous batching.** When a game ends, a new one starts in its slot immediately, so batches stay full until the last games drain.
3. **Leaf parallelism with virtual loss.** `--leaves-per-round K` lets each tree contribute K leaves per round (pessimistic virtual loss
   steers the K selections apart), for batches of games x K when few games are running. Tests check that no virtual loss is left behind.

Network evaluation is behind a small `Evaluator` interface (`evaluator.py`), with optional bf16 autocast on CUDA (`--amp`).
`python -m chessrl bench` measures forward throughput by batch size and self-play throughput by number of concurrent games on whatever
device you run it on, including the average batch actually sent to the network.

Not done: tree reuse between moves, multi-process self-play, and a separate inference server.

## Measuring strength

`evaluate.py` plays matches with colours alternated in pairs that share a random 4-ply opening, and reports W/D/L, score, and an Elo
difference with a ~95% interval (normal approximation of the trinomial game outcomes). Opponents (`players.py`):

| Opponent | What it is |
|----------|-----------|
| `random` | uniformly random legal moves |
| `material1` | one-move lookahead on material, takes mates in one |
| `material2` | two-ply material search (sees recaptures), takes mates |
| `stockfish` | Stockfish over UCI, strength-limited with `UCI_Elo` (needs the binary on your machine) |
| a checkpoint path | another trained net, to compare iterations |

Elo here is relative to the named opponent and says nothing absolute unless the opponent is calibrated (Stockfish's `UCI_Elo` is only
roughly comparable to human or rating-list scales). Matches use the same search budget (`--sims`) for the agent as stated in the output you record.

## Usage

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

pytest                                            # 38 tests, CPU only, ~10 s

python -m chessrl bench                           # throughput on this machine
python -m chessrl train --channels 64 --blocks 4 --iterations 100 \
    --games-per-iter 64 --concurrent-games 64 --sims 200 --out-dir runs/a
python -m chessrl eval --checkpoint runs/a/latest.pt --opponent material2 --games 40 --sims 200
python -m chessrl eval --checkpoint runs/a/latest.pt --opponent stockfish --stockfish-path /usr/games/stockfish --stockfish-elo 1350
```

Training writes `runs/<name>/metrics.jsonl` (one row per iteration: results mix, game length, self-play positions/s, mean network
batch, losses, periodic score against random), `config.json` and `latest.pt`.

## What is and is not verified

* Verified by tests: move encoding is collision-free and colour-symmetric over thousands of positions including castling, en passant and
  all promotion types; search finds mates and wins material with a material-based value function; per-game results do not depend on what else is in the batch;
  virtual loss is fully undone; self-play targets have the correct sign for each side; the trainer can fit a fixed batch and masks illegal moves; the whole
  train/evaluate loop runs end to end.
* The GPU code path (device selection, bf16 autocast) has **not** been run on a GPU. Everything here was developed and run on CPU.
* Playing strength depends heavily on the compute you give it (network size, simulations per move, games). Short CPU runs only show that the
  loop works; record your own `bench` and `eval` output for any claim about speed or strength.
