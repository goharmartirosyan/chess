#!/usr/bin/env python3
"""Step 2: depth-18 MultiPV 5 after each mistake, DEVELOPMENT DATA ONLY.

Requires: pip install python-chess pandas
Requires an installed Stockfish engine: --stockfish /path/to/stockfish

The input CSV must contain fen_before and uci, and a stable mistake identifier.
For each row, this runs a brand-new analysis AFTER the actual mistake, with the
opponent to move, BEFORE considering their real reply. Each engine instance is
used by just one process. The evaluator is deterministic at fixed engine settings
within a given Stockfish version; record that version for reproducibility.
"""
import argparse
import concurrent.futures as cf
import csv
import json
import math
import os
from pathlib import Path
import sys

import pandas as pd

K = 0.00368208  # recovered from the original dataset's cp-to-win conversion
RESULT_COLUMNS = [
    'multipv_good_reply_count_10pp',
    'multipv_best_second_gap_pp',
    'multipv_lines_returned',
    'multipv_top5_uci_json',
    'multipv_top5_win_chance_json',
    'multipv_top5_score_cp_json',
    'multipv_error',
]
KEY_COLUMNS = ['source_pgn', 'game_index', 'ply']
_worker_engine = None
_worker_depth = None
_worker_hash = None


def win_chance(cp: int) -> float:
    """Opponent's win chance, 0-100, same logistic mapping as source dataset."""
    val = max(-100.0, min(100.0, -K * cp))
    return 100.0 / (1.0 + math.exp(val))


def init_worker(engine_path: str, depth: int, hash_mb: int):
    global _worker_engine, _worker_depth, _worker_hash
    import chess.engine
    _worker_engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    _worker_depth = depth
    _worker_hash = hash_mb
    opts = _worker_engine.options
    config = {'Threads': 1}
    if 'Hash' in opts:
        config['Hash'] = hash_mb
    _worker_engine.configure(config)


def analyze_one(record):
    import chess
    import chess.engine
    row_id, fen, uci = record
    output = {'_row_id': row_id}
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f'Illegal mistake move {uci} for FEN')
        board.push(move)
        if board.is_game_over():
            # No legal reply means no punishable candidate; leave features missing
            raise ValueError('Position is terminal immediately after mistake')
        analyses = _worker_engine.analyse(
            board,
            chess.engine.Limit(depth=_worker_depth),
            multipv=min(5, board.legal_moves.count()),
            info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
        )
        # score is relative to side to move (opponent), not original mistake-maker
        lines = []
        for a in analyses:
            if not a.get('pv') or a.get('score') is None:
                continue
            score = a['score'].pov(board.turn)
            cp = score.score(mate_score=100000)
            if cp is None:
                continue
            lines.append((a['pv'][0].uci(), int(cp), win_chance(cp)))
        # Explicit ranking by win chance; for mate values ties are possible.
        lines.sort(key=lambda r: r[2], reverse=True)
        if not lines:
            raise ValueError('Stockfish returned no valid analysis lines')
        best = lines[0][2]
        output.update({
            'multipv_good_reply_count_10pp': sum((best - p) <= 10.0 + 1e-10 for _, _, p in lines),
            'multipv_best_second_gap_pp': (best - lines[1][2]) if len(lines) >= 2 else None,
            'multipv_lines_returned': len(lines),
            'multipv_top5_uci_json': json.dumps([x[0] for x in lines]),
            'multipv_top5_win_chance_json': json.dumps([round(x[2], 8) for x in lines]),
            'multipv_top5_score_cp_json': json.dumps([x[1] for x in lines]),
            'multipv_error': '',
        })
    except Exception as exc:
        output.update({c: None for c in RESULT_COLUMNS if c != 'multipv_error'})
        output['multipv_error'] = f'{type(exc).__name__}: {exc}'[:250]
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True, help='Five-archive development CSV only')
    ap.add_argument('--stockfish', required=True, help='Path to Stockfish binary')
    ap.add_argument('--output-dir', default='multipv_step2_results')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--depth', type=int, default=18)
    ap.add_argument('--hash-mb', type=int, default=64)
    ap.add_argument('--limit', type=int, help='Optional smoke-test N rows; omit for full run')
    ap.add_argument('--batch', type=int, default=100, help='Rows per checkpoint write')
    args = ap.parse_args()
    if args.workers < 1 or args.depth < 1 or args.batch < 1:
        ap.error('workers, depth and batch must be positive')
    source = Path(args.input).expanduser().resolve()
    engine_path = Path(args.stockfish).expanduser().resolve()
    if not source.is_file():
        ap.error(f'Missing input: {source}')
    if not engine_path.is_file():
        ap.error(f'Missing Stockfish executable: {engine_path}')
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows_file = outdir / 'multipv_raw_results.csv'
    enriched_file = outdir / 'development_with_multipv.csv'
    errors_file = outdir / 'multipv_failures.csv'
    df = pd.read_csv(source, low_memory=False)
    needed = ['fen_before', 'uci', *KEY_COLUMNS]
    missing = [c for c in needed if c not in df]
    if missing:
        ap.error('Input missing columns: ' + ', '.join(missing))
    # Protective guard: new/unseen archive may never enter development feature selection.
    archives = sorted(set(df['source_pgn'].dropna().astype(str)))
    expected = {
        'lichess_db_standard_rated_2017-04.pgn.zst',
        'lichess_db_standard_rated_2017-06.pgn.zst',
        'lichess_db_standard_rated_2017-07.pgn.zst',
        'lichess_db_standard_rated_2018-02.pgn.zst',
        'lichess_db_standard_rated_2019-02.pgn.zst',
    }
    if not set(archives).issubset(expected):
        ap.error('Input contains an archive outside the five development archives; refusing to analyze')
    if df.duplicated(KEY_COLUMNS).any():
        ap.error('Duplicate mistake identifiers; deduplicate before analysis')
    df['_row_id'] = range(len(df))
    if args.limit:
        df = df.iloc[:args.limit].copy()
    # Store raw results as append-only checkpoints and only one result per row.
    if rows_file.exists():
        prior = pd.read_csv(rows_file, low_memory=False)
        if not set(['_row_id', *RESULT_COLUMNS]).issubset(prior.columns):
            ap.error('Existing checkpoint schema differs; choose another output directory')
        done = set(prior['_row_id'].astype(int))
    else:
        done = set()
    records = [
        (int(r['_row_id']), str(r['fen_before']), str(r['uci']))
        for _, r in df.iterrows() if int(r['_row_id']) not in done
    ]
    print(f'Development rows: {len(df)}, already processed: {len(done)}, remaining: {len(records)}', flush=True)
    print('Archives:', ', '.join(archives), flush=True)
    print(f'Engine: {engine_path} | MultiPV 5 | depth {args.depth} | workers {args.workers}', flush=True)
    if records:
        # Map keeps input order; each separate worker owns a single engine process.
        with cf.ProcessPoolExecutor(
            max_workers=args.workers, initializer=init_worker,
            initargs=(str(engine_path), args.depth, args.hash_mb),
        ) as pool:
            with rows_file.open('a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['_row_id', *RESULT_COLUMNS])
                if f.tell() == 0:
                    writer.writeheader()
                for idx, result in enumerate(pool.map(analyze_one, records, chunksize=1), 1):
                    writer.writerow(result)
                    if idx % args.batch == 0:
                        f.flush()
                        os.fsync(f.fileno())
                        print(f'Processed {idx}/{len(records)} new positions', flush=True)
                f.flush()
                os.fsync(f.fileno())
    # Merge without changing any pre-existing feature/target, maintain CSV row order.
    raw = pd.read_csv(rows_file, low_memory=False).drop_duplicates('_row_id', keep='first')
    output = df.merge(raw, on='_row_id', how='left', validate='one_to_one', sort=False)
    output = output.drop(columns=['_row_id'])
    output.to_csv(enriched_file, index=False)
    failures = output[output['multipv_error'].notna() & output['multipv_error'].ne('')]
    failures.to_csv(errors_file, index=False)
    print('Rows saved:', len(output), 'errors:', len(failures), flush=True)
    print('Features CSV:', enriched_file, flush=True)
    print('Raw lines + checkpoints:', rows_file, flush=True)
    print('Failure review:', errors_file, flush=True)
    if args.limit:
        print('SMOKE TEST ONLY: this output contains only --limit rows, not full development data.', flush=True)

if __name__ == '__main__':
    main()
