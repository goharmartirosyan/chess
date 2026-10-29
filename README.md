# Chess Mistake Exploitation Dataset

This repository builds a research dataset for the question:

> Given a chess position, an engine-identified mistake, and immediate human game
> context, can we predict whether the opponent exploits the mistake and how much
> of the disadvantage persists over the next few moves?

The target is not objective move quality. Stockfish supplies that. The target is
human exploitation and persistence after the mistake.

## Input

The extractor streams complete Lichess PGN games from `.pgn` or `.pgn.zst`
without loading the full archive into memory.

The default input is:

```bash
lichess_db_standard_rated_2015-04.pgn.zst
```

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
brew install stockfish zstd
```

## Method

For every mainline move, `scripts/extract_bad_moves.py`:

1. Evaluates the position before the move with Stockfish.
2. Records Stockfish's best move.
3. Plays the human move.
4. Evaluates the resulting position from the original mover's perspective.
5. Converts centipawns to win chance using:

```text
win_chance = 100.0 / (1.0 + exp(-0.00368208 * score_cp))
```

6. Filters candidate mistakes primarily by configurable win-chance drop.
7. Evaluates the actual human continuation at 2, 4, 6, and 10 plies.
8. Constructs persistence targets from the original mistake-maker's perspective.

Mate scores are converted to a fixed bounded value, defaulting to `+/-1000 cp`.
The value is configurable with `--mate-score-cp`.

## Run

For an interactive version, open `extract_bad_moves.ipynb` and run the cells in
order. It includes the extraction code and a final **Run your extraction** cell
where you can edit the command options, including `--start-game`.

Fast smoke test:

```bash
python3 scripts/extract_bad_moves.py \
  --max-games 25 \
  --depth 8 \
  --multipv 1 \
  --skip-horizon-mistake-detection \
  --output bad_moves_smoke.csv
```

Research sample:

```bash
python3 scripts/extract_bad_moves.py \
  --max-games 3000 \
  --depth 18 \
  --min-win-loss 5 \
  --win-before-min 10 \
  --win-before-max 90 \
  --output bad_moves_depth18.csv
```

Combine multiple archives into one output file by repeating `--pgn`. Files are
streamed one after the other; `--max-games` is the combined game limit and the
output's `source_pgn` column identifies the source archive for every row.

```bash
python3 scripts/extract_bad_moves.py \
  --pgn lichess_2024_01.pgn.zst \
  --pgn lichess_2024_02.pgn.zst \
  --max-games 5000 \
  --depth 18 \
  --output combined_bad_moves.csv
```

Start from the 50th game, skipping the first 49 without engine analysis:

```bash
python3 scripts/extract_bad_moves.py \
  --pgn lichess_db_standard_rated_2017-04.pgn.zst \
  --start-game 50 \
  --max-games 25 \
  --output bad_moves_from_50.csv
```

This analyzes games 50–74. `--max-games` counts games analyzed after skipping;
`game_index` retains the original numbering. With multiple `--pgn` inputs,
`--start-game` counts across the archives in the order supplied. Compressed
archives still need to be decompressed sequentially to reach the starting game.

Run built-in perspective and mate sanity checks:

```bash
python3 scripts/extract_bad_moves.py --self-test
```

## Analyze Feature Distributions

After extraction, summarize the six-ply persistence targets and their descriptive
relationships to Elo, clocks, mistake severity, position complexity, phase, and
time control:

```bash
python3 scripts/analyze_feature_distributions.py \
  --input bad_moves_depth18.csv \
  --output feature_distributions.html
```

This creates one standalone HTML file with distribution tables, histograms,
correlation bars, scatter plots, numeric quintile summaries, and grouped
summaries for game phase, time control, tactical flags, player strength, clocks,
evaluations, material, mobility, and Stockfish complexity features.
By default, the analysis includes only rows with `initial_damage > 20`. Change
this reporting subset with `--min-initial-damage VALUE`.

## Important Options

- `--depth 18`: fixed Stockfish depth. Use `--nodes` or `--time` instead if desired.
- `--min-win-loss 5`: primary mistake-selection threshold in win-chance points.
- `--min-cp-loss 0`: optional centipawn-loss floor; centipawn loss is always stored.
- `--win-before-min 10 --win-before-max 90`: excludes already-decided positions.
- `--multipv 3`: enables candidate-move complexity features.
- `--reasonable-move-cp 50`: threshold for counting reasonable MultiPV moves.
- `--shallow-depth N`: adds shallow-vs-deep best-move agreement features.
- `--skip-horizon-mistake-detection`: disables expensive future-mistake flags.
- `--use-terminal-result-targets`: when a horizon is not reached, optionally uses
  the final game result as a target substitute. By default this is off.

## Output

One row represents one selected bad move. CSV and JSON Lines are supported; use a
`.csv` output extension for CSV.

Rows include identifiers and metadata:

- `game_index`, `event`, `site`, `date`, `white`, `black`, `result`
- `white_elo`, `black_elo`, `player_elo`, `opponent_elo`, `rating_difference`
- `time_control`, `ply`, `move_number`, `side`
- `san`, `uci`, `best_move_san`, `best_move_uci`

Core mistake fields:

- `eval_before_cp`, `eval_after_cp`
- `centipawn_loss`
- `win_before`, `win_after`, `win_chance_drop`
- `severity_label`: `blunder`, `mistake`, `inaccuracy`, or `small_loss`

Immediate model-input features include:

- `fen_before`
- previous move/context fields
- clock fields and missing-clock flags
- game phase, material balance, piece counts, pawn count
- mobility, legal move count, branching factor
- hanging-piece counts, king attackers
- capture/check/promotion flags
- optional MultiPV and shallow/deep agreement features

Target-construction fields include:

- `eval_after_2ply_cp`, `eval_after_4ply_cp`, `eval_after_6ply_cp`,
  `eval_after_10ply_cp`
- `win_after_2ply`, `win_after_4ply`, `win_after_6ply`, `win_after_10ply`
- `remaining_damage_*`
- `persistence_ratio_*`
- `persistence_ratio_*_clipped`
- `persistence_category_*`

For each horizon, the script also stores:

- `horizon_*_reached`
- `game_ended_before_*`
- `terminal_result_used_for_*`
- `mate_score_used_after_*`
- flags for another large mistake by either player, the original player, or the opponent

Future fields, final result, and next-move context are included for analysis and
target construction. Do not use them as model inputs.
