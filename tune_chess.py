#!/usr/bin/env python3
"""Development-only chess regression tuning + CV, with safe game-grouped splits.

Install: python3 -m pip install pandas numpy scipy scikit-learn lightgbm
Run tuning: python3 tune_chess.py --input development_with_multipv.csv --config config.json --out my_tuning
Run a chosen config: python3 tune_chess.py --input development_with_multipv.csv --config config.json --out my_cv --mode evaluate --chosen my_tuning/selected.json

Warning: evaluation on the same development archives used for tuning is exploratory,
not independent confirmation. Keep March 2025 sealed until all choices are frozen.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor

EXPECTED_ARCHIVES = {
    'lichess_db_standard_rated_2017-04.pgn.zst',
    'lichess_db_standard_rated_2017-06.pgn.zst',
    'lichess_db_standard_rated_2017-07.pgn.zst',
    'lichess_db_standard_rated_2018-02.pgn.zst',
    'lichess_db_standard_rated_2019-02.pgn.zst',
}
EVAL = ['eval_before_cp', 'initial_damage']
RATINGS = ['player_elo', 'opponent_elo']
CONTEXT = ['move_number', 'player_clock_before_move', 'opponent_clock_before_move', 'initial_time_seconds', 'increment_seconds']
CHESS = ['material_balance_cp', 'number_of_pieces', 'hanging_pieces', 'opponent_hanging_pieces',
         'king_attackers', 'opponent_king_attackers', 'reasonable_candidate_moves',
         'eval_gap_best_second_cp', 'in_check', 'gives_check', 'is_capture', 'is_promotion', 'game_phase']
MULTIPV = ['multipv_good_reply_count_10pp', 'multipv_best_second_gap_pp']
SETS = {
    'evaluation': EVAL,
    'evaluation_ratings': EVAL + RATINGS,
    'context': EVAL + RATINGS + CONTEXT,
    'context_multipv': EVAL + RATINGS + CONTEXT + MULTIPV,
    'full': EVAL + RATINGS + CONTEXT + CHESS,
    'full_multipv': EVAL + RATINGS + CONTEXT + CHESS + MULTIPV,
}


def pipeline(features, params, seed, jobs):
    numeric = [f for f in features if f != 'game_phase']
    categorical = [f for f in features if f == 'game_phase']
    transformers = [('numeric', SimpleImputer(strategy='median', add_indicator=True), numeric)]
    if categorical:
        transformers.append(('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical))
    prep = ColumnTransformer(transformers)
    return Pipeline([('prep', prep), ('model', LGBMRegressor(**params, n_jobs=jobs, random_state=seed))])


def read_dev(path):
    df = pd.read_csv(path, low_memory=False)
    actual = set(df['source_pgn'].dropna().astype(str).map(lambda s: Path(s).name).unique())
    if actual != EXPECTED_ARCHIVES:
        raise ValueError(f'This script is for the FIVE development archives only! Found: {sorted(actual)}')
    if df.duplicated(['source_pgn', 'game_index', 'ply']).any():
        raise ValueError('Duplicate mistake identifiers found; use the deduplicated dataset.')
    if not ((df['initial_damage'] > 20) & df['win_before'].between(10, 90)).all():
        raise ValueError('Data failed the pre-registered selection thresholds.')
    game = df['site'].fillna('').astype(str)
    fallback = df['source_pgn'].astype(str) + '::' + df['game_index'].astype(str)
    game = game.where(game.str.strip() != '', fallback)
    return df, game


def get_folds(df, groups, count, seed):
    if count < 2:
        raise ValueError('Use at least two folds.')
    folds = list(GroupKFold(n_splits=count, shuffle=True, random_state=seed).split(df, groups=groups))
    for train, test in folds:
        assert set(groups.iloc[train]).isdisjoint(set(groups.iloc[test]))
    return folds


def cv(df, groups, folds, horizon, features, params, seed, jobs, keep_predictions=False):
    target = f'change_in_damage_{horizon}ply'
    y = pd.to_numeric(df[target], errors='coerce')
    maes = []
    prediction_rows = []
    for fold_id, (tr0, te0) in enumerate(folds, 1):
        tr = tr0[y.iloc[tr0].notna().to_numpy()]
        te = te0[y.iloc[te0].notna().to_numpy()]
        if not len(tr) or not len(te):
            raise ValueError(f'No nonmissing targets for horizon={horizon}, fold={fold_id}')
        model = pipeline(features, params, seed, jobs)
        model.fit(df.iloc[tr][features], y.iloc[tr])
        pred = model.predict(df.iloc[te][features])
        true = y.iloc[te].to_numpy()
        maes.append(mean_absolute_error(true, pred))
        if keep_predictions:
            for rid, game_id, actual, p in zip(te, groups.iloc[te], true, pred):
                prediction_rows.append(dict(row_index=int(rid), game=str(game_id), fold=fold_id,
                                            horizon=horizon, actual=float(actual), prediction=float(p)))
    return float(np.mean(maes)), prediction_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--config', type=Path, default=Path('config.json'))
    ap.add_argument('--out', type=Path, default=Path('my_tuning'))
    ap.add_argument('--mode', choices=['tune', 'evaluate'], default='tune')
    ap.add_argument('--chosen', type=Path, help='selected.json generated by tune mode')
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    df, groups = read_dev(args.input)
    args.out.mkdir(parents=True, exist_ok=True)
    horizons = [int(h) for h in config['horizons']]
    features_names = config['feature_sets']
    for fs in features_names:
        if fs not in SETS:
            raise ValueError(f'Unknown feature set: {fs}. Valid: {list(SETS)}')
        absent = [x for x in SETS[fs] if x not in df.columns]
        if absent:
            raise ValueError(f'Missing columns for {fs}: {absent}')
    folds = get_folds(df, groups, int(config['folds']), int(config['seed']))
    base = config['base_params'].copy()
    records = []
    if args.mode == 'tune':
        search_space = config['search_space']
        total_combos = int(np.prod([len(v) for v in search_space.values()]))
        candidates = list(ParameterSampler(search_space, n_iter=min(int(config['max_configs']), total_combos),
                                           random_state=int(config['seed'])))
        selected = {'method': 'development-only game-grouped CV selection; NOT independent test results',
                    'seed': config['seed'], 'folds': config['folds'], 'horizons': {}}
        for h in horizons:
            best = None
            for fs in features_names:
                for j, overrides in enumerate(candidates, 1):
                    params = {**base, **overrides}
                    mae, _ = cv(df, groups, folds, h, SETS[fs], params,
                                int(config['seed']), int(config['n_jobs']))
                    row = dict(horizon=h, feature_set=fs, candidate=j, mean_fold_mae=mae,
                               params=json.dumps(params, sort_keys=True))
                    records.append(row)
                    if best is None or mae < best['selection_mae']:
                        best = dict(feature_set=fs, features=SETS[fs], params=params, selection_mae=mae)
                    print(f'h={h:2d}  {fs:17s}  {j}/{len(candidates)}  MAE={mae:.4f}', flush=True)
                    pd.DataFrame(records).to_csv(args.out / 'search_results.csv', index=False)
            selected['horizons'][str(h)] = best
            print(f'SELECTED h={h}: {best["feature_set"]}, MAE={best["selection_mae"]:.4f}', flush=True)
            (args.out / 'selected.json').write_text(json.dumps(selected, indent=2))
        print('Saved:', args.out / 'selected.json')
    else:
        if not args.chosen:
            ap.error('--mode evaluate requires --chosen path/to/selected.json')
        chosen = json.loads(args.chosen.read_text())
        summaries, predictions = [], []
        for h in horizons:
            entry = chosen['horizons'][str(h)]
            fs = entry['feature_set']
            features = entry['features']
            params = entry['params']
            if any(c not in df.columns for c in features):
                raise ValueError('Chosen configuration references unavailable features')
            mae, pred_rows = cv(df, groups, folds, h, features, params,
                                int(config['seed']), int(config['n_jobs']), keep_predictions=True)
            for p in pred_rows:
                p['model'] = fs
            predictions.extend(pred_rows)
            sub = [r for r in pred_rows if r['horizon'] == h]
            y = np.array([r['actual'] for r in sub]); pred = np.array([r['prediction'] for r in sub])
            baseline = float(np.mean(np.abs(y)))
            rho = spearmanr(y, pred).statistic if np.std(pred) > 1e-12 else np.nan
            summaries.append(dict(horizon=h, feature_set=fs, selected_mae=mae,
                                  pooled_mae=mean_absolute_error(y, pred), no_change_mae=baseline,
                                  pooled_improvement=baseline - mean_absolute_error(y, pred),
                                  pooled_improvement_pct=100*(baseline-mean_absolute_error(y,pred))/baseline,
                                  pooled_spearman=float(rho)))
            print(f'h={h}: LightGBM MAE={mae:.3f}; no-change pooled MAE={baseline:.3f}',flush=True)
        pd.DataFrame(summaries).to_csv(args.out / 'cv_summary.csv', index=False)
        pd.DataFrame(predictions).to_csv(args.out / 'cv_predictions.csv', index=False)
        print('Saved:', args.out / 'cv_summary.csv')


if __name__ == '__main__':
    main()
