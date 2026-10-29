#!/usr/bin/env python3
"""Build a chess mistake exploitation dataset from streamed PGN games."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable

import chess
import chess.engine
import chess.pgn


FUTURE_PLIES = (2, 4, 6, 10)
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}


@dataclass
class Config:
    future_plies: tuple[int, ...]
    mate_score_cp: int
    min_win_loss: float
    min_cp_loss: int
    win_before_min: float
    win_before_max: float
    context_plies: int
    persistence_epsilon: float
    persistence_clip_min: float
    persistence_clip_max: float
    mostly_recovered_max: float
    partially_persistent_max: float
    strongly_persistent_max: float
    horizon_mistake_win_loss: float
    horizon_mistake_cp_loss: int
    detect_horizon_mistakes: bool
    multipv: int
    reasonable_move_cp: int
    shallow_depth: int | None
    use_terminal_result_targets: bool


@dataclass
class MoveRecord:
    ply: int
    move: chess.Move
    san: str
    uci: str
    mover: chess.Color
    node: chess.pgn.GameNode
    clock_before_player: float | None
    clock_before_opponent: float | None
    clock_after: float | None


@dataclass
class DatasetRow:
    game_index: int
    source_pgn: str
    event: str
    site: str
    date: str
    white: str
    black: str
    result: str
    white_elo: int | None
    black_elo: int | None
    player_elo: int | None
    opponent_elo: int | None
    rating_difference: int | None
    time_control: str
    ply: int
    move_number: int
    side: str
    san: str
    uci: str
    best_move_san: str
    best_move_uci: str
    eval_before_cp: int
    eval_after_cp: int
    centipawn_loss: int
    win_before: float
    win_after: float
    win_chance_drop: float
    severity_label: str
    fen_before: str
    previous_moves_san: str
    next_moves_san: str
    move_context_san: str
    previous_move_san: str
    previous_move_uci: str
    player_clock_before_move: float | None
    opponent_clock_before_move: float | None
    clock_difference: float | None
    player_clock_missing: bool
    opponent_clock_missing: bool
    initial_time_seconds: int | None
    increment_seconds: int | None
    player_clock_pct_initial: float | None
    opponent_clock_pct_initial: float | None
    game_phase: str
    material_balance_cp: int
    number_of_pieces: int
    number_of_pawns: int
    mobility: int
    opponent_mobility: int
    legal_moves: int
    branching_factor: int
    hanging_pieces: int
    opponent_hanging_pieces: int
    king_attackers: int
    opponent_king_attackers: int
    in_check: bool
    gives_check: bool
    is_capture: bool
    is_promotion: bool
    reasonable_candidate_moves: int | None
    eval_gap_best_second_cp: int | None
    eval_gap_best_third_cp: int | None
    shallow_best_move_uci: str
    shallow_deep_best_agree: bool | None
    punishment_only_deep: bool | None
    eval_after_2ply_cp: int | None
    eval_after_4ply_cp: int | None
    eval_after_6ply_cp: int | None
    eval_after_10ply_cp: int | None
    win_after_2ply: float | None
    win_after_4ply: float | None
    win_after_6ply: float | None
    win_after_10ply: float | None
    horizon_2ply_reached: bool
    horizon_4ply_reached: bool
    horizon_6ply_reached: bool
    horizon_10ply_reached: bool
    game_ended_before_2ply: bool
    game_ended_before_4ply: bool
    game_ended_before_6ply: bool
    game_ended_before_10ply: bool
    terminal_result_used_for_2ply: bool
    terminal_result_used_for_4ply: bool
    terminal_result_used_for_6ply: bool
    terminal_result_used_for_10ply: bool
    mate_score_used_after: bool
    mate_score_used_after_2ply: bool
    mate_score_used_after_4ply: bool
    mate_score_used_after_6ply: bool
    mate_score_used_after_10ply: bool
    initial_damage: float
    remaining_damage_2ply: float | None
    remaining_damage_4ply: float | None
    remaining_damage_6ply: float | None
    remaining_damage_10ply: float | None
    persistence_ratio_2ply: float | None
    persistence_ratio_4ply: float | None
    persistence_ratio_6ply: float | None
    persistence_ratio_10ply: float | None
    persistence_ratio_2ply_clipped: float | None
    persistence_ratio_4ply_clipped: float | None
    persistence_ratio_6ply_clipped: float | None
    persistence_ratio_10ply_clipped: float | None
    persistence_category_2ply: str
    persistence_category_4ply: str
    persistence_category_6ply: str
    persistence_category_10ply: str
    another_large_mistake_within_2ply: bool
    another_large_mistake_within_4ply: bool
    another_large_mistake_within_6ply: bool
    another_large_mistake_within_10ply: bool
    original_player_large_mistake_within_2ply: bool
    original_player_large_mistake_within_4ply: bool
    original_player_large_mistake_within_6ply: bool
    original_player_large_mistake_within_10ply: bool
    opponent_large_mistake_within_2ply: bool
    opponent_large_mistake_within_4ply: bool
    opponent_large_mistake_within_6ply: bool
    opponent_large_mistake_within_10ply: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream PGN/PGN.zst games and build a bad-move exploitation dataset."
    )
    parser.add_argument(
        "--pgn",
        action="append",
        default=None,
        help="Input .pgn or .pgn.zst file. Repeat this option to combine archives.",
    )
    parser.add_argument("--engine", default=shutil.which("stockfish") or "stockfish")
    parser.add_argument("--output", default="bad_moves.jsonl")
    parser.add_argument("--max-games", type=int, default=25)
    parser.add_argument(
        "--start-game",
        type=int,
        default=1,
        help="First game to analyze (1-based, across all inputs); earlier games are skipped.",
    )
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--nodes", type=int, default=None)
    parser.add_argument("--time", type=float, default=None)
    parser.add_argument("--mate-score-cp", type=int, default=1000)
    parser.add_argument("--min-win-loss", type=float, default=5.0)
    parser.add_argument("--min-cp-loss", type=int, default=0)
    parser.add_argument("--win-before-min", type=float, default=10.0)
    parser.add_argument("--win-before-max", type=float, default=90.0)
    parser.add_argument("--context-plies", type=int, default=5)
    parser.add_argument("--persistence-epsilon", type=float, default=1e-6)
    parser.add_argument("--persistence-clip-min", type=float, default=0.0)
    parser.add_argument("--persistence-clip-max", type=float, default=2.0)
    parser.add_argument("--mostly-recovered-max", type=float, default=0.25)
    parser.add_argument("--partially-persistent-max", type=float, default=0.75)
    parser.add_argument("--strongly-persistent-max", type=float, default=1.0)
    parser.add_argument("--horizon-mistake-win-loss", type=float, default=5.0)
    parser.add_argument("--horizon-mistake-cp-loss", type=int, default=100)
    parser.add_argument("--skip-horizon-mistake-detection", action="store_true")
    parser.add_argument("--multipv", type=int, default=3)
    parser.add_argument("--reasonable-move-cp", type=int, default=50)
    parser.add_argument("--shallow-depth", type=int, default=None)
    parser.add_argument(
        "--use-terminal-result-targets",
        action="store_true",
        help="Use final result as a win chance substitute when a future horizon is not reached.",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.start_game < 1:
        parser.error("--start-game must be at least 1")
    return args


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        future_plies=FUTURE_PLIES,
        mate_score_cp=args.mate_score_cp,
        min_win_loss=args.min_win_loss,
        min_cp_loss=args.min_cp_loss,
        win_before_min=args.win_before_min,
        win_before_max=args.win_before_max,
        context_plies=args.context_plies,
        persistence_epsilon=args.persistence_epsilon,
        persistence_clip_min=args.persistence_clip_min,
        persistence_clip_max=args.persistence_clip_max,
        mostly_recovered_max=args.mostly_recovered_max,
        partially_persistent_max=args.partially_persistent_max,
        strongly_persistent_max=args.strongly_persistent_max,
        horizon_mistake_win_loss=args.horizon_mistake_win_loss,
        horizon_mistake_cp_loss=args.horizon_mistake_cp_loss,
        detect_horizon_mistakes=not args.skip_horizon_mistake_detection,
        multipv=max(1, args.multipv),
        reasonable_move_cp=args.reasonable_move_cp,
        shallow_depth=args.shallow_depth,
        use_terminal_result_targets=args.use_terminal_result_targets,
    )


def open_pgn_stream(path: Path) -> tuple[io.TextIOBase, subprocess.Popen[bytes] | None]:
    if path.suffix == ".zst":
        if not shutil.which("zstd"):
            raise SystemExit("zstd is required to stream .zst PGN files.")
        proc = subprocess.Popen(["zstd", "-dc", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.stdout is None:
            raise SystemExit("Could not open zstd output stream.")
        return io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace"), proc
    return path.open("r", encoding="utf-8", errors="replace"), None


def limit_from_args(args: argparse.Namespace) -> chess.engine.Limit:
    if args.time is not None:
        return chess.engine.Limit(time=args.time)
    if args.nodes is not None:
        return chess.engine.Limit(nodes=args.nodes)
    return chess.engine.Limit(depth=args.depth)


def cp_score_and_mate(info: dict, color: chess.Color, mate_score_cp: int) -> tuple[int, bool]:
    pov_score = info["score"].pov(color)
    mate = pov_score.mate()
    if mate is not None:
        return (mate_score_cp if mate > 0 else -mate_score_cp), True
    return int(pov_score.score()), False


def winning_chance(score_cp: float) -> float:
    return 100.0 / (1.0 + math.exp(-0.00368208 * score_cp))


def terminal_win_chance(result: str, color: chess.Color) -> float | None:
    if result == "1/2-1/2":
        return 50.0
    if result == "1-0":
        return 100.0 if color == chess.WHITE else 0.0
    if result == "0-1":
        return 100.0 if color == chess.BLACK else 0.0
    return None


def parse_int_header(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"\d+", value)
    return int(match.group()) if match else None


def parse_time_control(value: str) -> tuple[int | None, int | None]:
    if not value or value == "-":
        return None, None
    first = value.split(":", 1)[0]
    if "+" in first:
        initial, increment = first.split("+", 1)
        return parse_int_header(initial), parse_int_header(increment)
    return parse_int_header(first), 0


def maybe_difference(left: float | int | None, right: float | int | None) -> float | int | None:
    if left is None or right is None:
        return None
    return left - right


def percent_of_initial(clock: float | None, initial: int | None) -> float | None:
    if clock is None or not initial:
        return None
    return 100.0 * clock / initial


def classify_loss(loss: int) -> str:
    if loss >= 300:
        return "blunder"
    if loss >= 100:
        return "mistake"
    if loss >= 50:
        return "inaccuracy"
    return "small_loss"


def classify_persistence(ratio: float | None, config: Config) -> str:
    if ratio is None:
        return ""
    if ratio < config.mostly_recovered_max:
        return "mostly_recovered"
    if ratio < config.partially_persistent_max:
        return "partially_persistent"
    if ratio <= config.strongly_persistent_max:
        return "strongly_persistent"
    return "worsened_further"


def clipped(value: float | None, config: Config) -> float | None:
    if value is None:
        return None
    return min(config.persistence_clip_max, max(config.persistence_clip_min, value))


def move_label(ply: int, san: str) -> str:
    move_number = (ply + 1) // 2
    return f"{move_number}. {san}" if ply % 2 == 1 else f"{move_number}... {san}"


def collect_moves(game: chess.pgn.Game) -> list[MoveRecord]:
    board = game.board()
    node = game
    records: list[MoveRecord] = []
    last_clock_after: dict[chess.Color, float | None] = {chess.WHITE: None, chess.BLACK: None}

    for ply, child in enumerate(game.mainline(), start=1):
        move = child.move
        if move is None:
            continue
        mover = board.turn
        san = board.san(move)
        clock_before_player = last_clock_after[mover]
        clock_before_opponent = last_clock_after[not mover]
        board.push(move)
        clock_after = child.clock()
        if clock_after is not None:
            last_clock_after[mover] = clock_after
        records.append(
            MoveRecord(
                ply=ply,
                move=move,
                san=san,
                uci=move.uci(),
                mover=mover,
                node=node,
                clock_before_player=clock_before_player,
                clock_before_opponent=clock_before_opponent,
                clock_after=clock_after,
            )
        )
        node = child
    return records


def context_fields(records: list[MoveRecord], bad_ply: int, context_plies: int) -> tuple[str, str, str]:
    bad_index = bad_ply - 1
    start = max(0, bad_index - context_plies)
    end = min(len(records), bad_index + context_plies + 1)
    before = [move_label(index + 1, records[index].san) for index in range(start, bad_index)]
    after = [move_label(index + 1, records[index].san) for index in range(bad_index + 1, end)]
    full_context = []
    for index in range(start, end):
        label = move_label(index + 1, records[index].san)
        full_context.append(f"*{label}*" if index == bad_index else label)
    return " ".join(before), " ".join(after), " ".join(full_context)


def board_after(records: list[MoveRecord], ply: int, start_board: chess.Board) -> chess.Board:
    board = start_board.copy(stack=False)
    for record in records[:ply]:
        board.push(record.move)
    return board


def best_move_fields(board: chess.Board, info: dict) -> tuple[str, str]:
    pv = info.get("pv") or []
    if not pv:
        return "", ""
    best_move = pv[0]
    return board.san(best_move), best_move.uci()


def material_balance(board: chess.Board, color: chess.Color) -> int:
    total = 0
    for piece_type, value in PIECE_VALUES.items():
        total += len(board.pieces(piece_type, color)) * value
        total -= len(board.pieces(piece_type, not color)) * value
    return total


def game_phase(board: chess.Board) -> str:
    non_pawn_material = 0
    for color in chess.COLORS:
        non_pawn_material += (
            len(board.pieces(chess.KNIGHT, color)) * 3
            + len(board.pieces(chess.BISHOP, color)) * 3
            + len(board.pieces(chess.ROOK, color)) * 5
            + len(board.pieces(chess.QUEEN, color)) * 9
        )
    if non_pawn_material >= 42:
        return "opening"
    if non_pawn_material >= 20:
        return "middlegame"
    return "endgame"


def mobility_for(board: chess.Board, color: chess.Color) -> int:
    copy = board.copy(stack=False)
    copy.turn = color
    return copy.legal_moves.count()


def hanging_pieces(board: chess.Board, color: chess.Color) -> int:
    count = 0
    for square in chess.SquareSet(board.occupied_co[color]):
        piece = board.piece_at(square)
        if piece and piece.piece_type == chess.KING:
            continue
        if board.is_attacked_by(not color, square) and not board.is_attacked_by(color, square):
            count += 1
    return count


def king_attackers(board: chess.Board, color: chess.Color) -> int:
    king_square = board.king(color)
    return 0 if king_square is None else len(board.attackers(not color, king_square))


def multipv_features(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    limit: chess.engine.Limit,
    config: Config,
    deep_best_uci: str,
) -> tuple[int | None, int | None, int | None, str, bool | None, bool | None]:
    reasonable = gap_second = gap_third = None
    shallow_best_uci = ""
    shallow_deep_agree = punishment_only_deep = None

    if config.multipv > 1:
        infos = engine.analyse(board, limit, multipv=config.multipv)
        if isinstance(infos, dict):
            infos = [infos]
        scored = []
        for info in infos:
            pv = info.get("pv") or []
            if not pv:
                continue
            score, _ = cp_score_and_mate(info, board.turn, config.mate_score_cp)
            scored.append((pv[0].uci(), score))
        scored.sort(key=lambda item: item[1], reverse=True)
        if scored:
            best_score = scored[0][1]
            reasonable = sum(1 for _, score in scored if best_score - score <= config.reasonable_move_cp)
            if len(scored) >= 2:
                gap_second = best_score - scored[1][1]
            if len(scored) >= 3:
                gap_third = best_score - scored[2][1]

    if config.shallow_depth is not None and config.shallow_depth > 0:
        shallow_info = engine.analyse(board, chess.engine.Limit(depth=config.shallow_depth))
        pv = shallow_info.get("pv") or []
        if pv:
            shallow_best_uci = pv[0].uci()
            shallow_deep_agree = shallow_best_uci == deep_best_uci
            punishment_only_deep = not shallow_deep_agree

    return reasonable, gap_second, gap_third, shallow_best_uci, shallow_deep_agree, punishment_only_deep


def future_eval_maps(
    records: list[MoveRecord],
    start_board: chess.Board,
    bad_ply: int,
    mover: chess.Color,
    result: str,
    engine: chess.engine.SimpleEngine,
    limit: chess.engine.Limit,
    config: Config,
) -> tuple[dict[int, int | None], dict[int, float | None], dict[int, bool], dict[int, bool], dict[int, bool]]:
    cp_by_horizon: dict[int, int | None] = {}
    win_by_horizon: dict[int, float | None] = {}
    reached: dict[int, bool] = {}
    terminal_used: dict[int, bool] = {}
    mate_used: dict[int, bool] = {}

    for horizon in config.future_plies:
        target_ply = bad_ply + horizon
        reached[horizon] = target_ply <= len(records)
        terminal_used[horizon] = False
        mate_used[horizon] = False
        if reached[horizon]:
            future_board = board_after(records, target_ply, start_board)
            info = engine.analyse(future_board, limit)
            cp, is_mate = cp_score_and_mate(info, mover, config.mate_score_cp)
            cp_by_horizon[horizon] = cp
            win_by_horizon[horizon] = winning_chance(cp)
            mate_used[horizon] = is_mate
            continue

        cp_by_horizon[horizon] = None
        win_by_horizon[horizon] = None
        if config.use_terminal_result_targets:
            terminal_win = terminal_win_chance(result, mover)
            if terminal_win is not None:
                win_by_horizon[horizon] = terminal_win
                terminal_used[horizon] = True

    return cp_by_horizon, win_by_horizon, reached, terminal_used, mate_used


def horizon_mistake_flags(
    records: list[MoveRecord],
    start_board: chess.Board,
    bad_ply: int,
    original_mover: chess.Color,
    engine: chess.engine.SimpleEngine,
    limit: chess.engine.Limit,
    config: Config,
) -> tuple[dict[int, bool], dict[int, bool], dict[int, bool]]:
    any_flags = {horizon: False for horizon in config.future_plies}
    original_flags = {horizon: False for horizon in config.future_plies}
    opponent_flags = {horizon: False for horizon in config.future_plies}
    if not config.detect_horizon_mistakes:
        return any_flags, original_flags, opponent_flags

    max_ply = min(len(records), bad_ply + max(config.future_plies))
    board = board_after(records, bad_ply, start_board)
    for index in range(bad_ply, max_ply):
        record = records[index]
        mover = board.turn
        before_info = engine.analyse(board, limit)
        before_cp, _ = cp_score_and_mate(before_info, mover, config.mate_score_cp)
        board.push(record.move)
        after_info = engine.analyse(board, limit)
        after_cp, _ = cp_score_and_mate(after_info, mover, config.mate_score_cp)
        loss = before_cp - after_cp
        win_loss = winning_chance(before_cp) - winning_chance(after_cp)
        if loss < config.horizon_mistake_cp_loss and win_loss < config.horizon_mistake_win_loss:
            continue
        plies_after = record.ply - bad_ply
        for horizon in config.future_plies:
            if plies_after <= horizon:
                any_flags[horizon] = True
                if mover == original_mover:
                    original_flags[horizon] = True
                else:
                    opponent_flags[horizon] = True
    return any_flags, original_flags, opponent_flags


def build_row(
    game: chess.pgn.Game,
    game_index: int,
    source_pgn: str,
    records: list[MoveRecord],
    record: MoveRecord,
    board: chess.Board,
    before_info: dict,
    eval_before: int,
    eval_after: int,
    mate_after: bool,
    engine: chess.engine.SimpleEngine,
    limit: chess.engine.Limit,
    config: Config,
) -> DatasetRow:
    headers = game.headers
    mover = record.mover
    white_elo = parse_int_header(headers.get("WhiteElo"))
    black_elo = parse_int_header(headers.get("BlackElo"))
    player_elo = white_elo if mover == chess.WHITE else black_elo
    opponent_elo = black_elo if mover == chess.WHITE else white_elo
    initial_time, increment = parse_time_control(headers.get("TimeControl", ""))
    best_san, best_uci = best_move_fields(board, before_info)
    legal_moves = board.legal_moves.count()
    board_after_move = board.copy(stack=False)
    board_after_move.push(record.move)
    future_cp, future_win, reached, terminal_used, future_mate = future_eval_maps(
        records, game.board(), record.ply, mover, headers.get("Result", ""), engine, limit, config
    )
    any_mistakes, original_mistakes, opponent_mistakes = horizon_mistake_flags(
        records, game.board(), record.ply, mover, engine, limit, config
    )
    win_before = winning_chance(eval_before)
    win_after = winning_chance(eval_after)
    initial_damage = win_before - win_after
    remaining = {
        horizon: (win_before - future_win[horizon]) if future_win[horizon] is not None else None
        for horizon in config.future_plies
    }
    ratios = {
        horizon: (
            remaining[horizon] / initial_damage
            if remaining[horizon] is not None and abs(initial_damage) > config.persistence_epsilon
            else None
        )
        for horizon in config.future_plies
    }
    previous_san, next_san, context_san = context_fields(records, record.ply, config.context_plies)
    previous_record = records[record.ply - 2] if record.ply >= 2 else None
    complexity = multipv_features(board, engine, limit, config, best_uci)

    return DatasetRow(
        game_index=game_index,
        source_pgn=source_pgn,
        event=headers.get("Event", ""),
        site=headers.get("Site", ""),
        date=headers.get("Date", ""),
        white=headers.get("White", ""),
        black=headers.get("Black", ""),
        result=headers.get("Result", ""),
        white_elo=white_elo,
        black_elo=black_elo,
        player_elo=player_elo,
        opponent_elo=opponent_elo,
        rating_difference=maybe_difference(player_elo, opponent_elo),
        time_control=headers.get("TimeControl", ""),
        ply=record.ply,
        move_number=(record.ply + 1) // 2,
        side="White" if mover == chess.WHITE else "Black",
        san=record.san,
        uci=record.uci,
        best_move_san=best_san,
        best_move_uci=best_uci,
        eval_before_cp=eval_before,
        eval_after_cp=eval_after,
        centipawn_loss=eval_before - eval_after,
        win_before=win_before,
        win_after=win_after,
        win_chance_drop=initial_damage,
        severity_label=classify_loss(eval_before - eval_after),
        fen_before=board.fen(),
        previous_moves_san=previous_san,
        next_moves_san=next_san,
        move_context_san=context_san,
        previous_move_san=previous_record.san if previous_record else "",
        previous_move_uci=previous_record.uci if previous_record else "",
        player_clock_before_move=record.clock_before_player,
        opponent_clock_before_move=record.clock_before_opponent,
        clock_difference=maybe_difference(record.clock_before_player, record.clock_before_opponent),
        player_clock_missing=record.clock_before_player is None,
        opponent_clock_missing=record.clock_before_opponent is None,
        initial_time_seconds=initial_time,
        increment_seconds=increment,
        player_clock_pct_initial=percent_of_initial(record.clock_before_player, initial_time),
        opponent_clock_pct_initial=percent_of_initial(record.clock_before_opponent, initial_time),
        game_phase=game_phase(board),
        material_balance_cp=material_balance(board, mover),
        number_of_pieces=len(board.piece_map()),
        number_of_pawns=len(board.pieces(chess.PAWN, chess.WHITE)) + len(board.pieces(chess.PAWN, chess.BLACK)),
        mobility=mobility_for(board, mover),
        opponent_mobility=mobility_for(board, not mover),
        legal_moves=legal_moves,
        branching_factor=legal_moves,
        hanging_pieces=hanging_pieces(board, mover),
        opponent_hanging_pieces=hanging_pieces(board, not mover),
        king_attackers=king_attackers(board, mover),
        opponent_king_attackers=king_attackers(board, not mover),
        in_check=board.is_check(),
        gives_check=board_after_move.is_check(),
        is_capture=board.is_capture(record.move),
        is_promotion=record.move.promotion is not None,
        reasonable_candidate_moves=complexity[0],
        eval_gap_best_second_cp=complexity[1],
        eval_gap_best_third_cp=complexity[2],
        shallow_best_move_uci=complexity[3],
        shallow_deep_best_agree=complexity[4],
        punishment_only_deep=complexity[5],
        eval_after_2ply_cp=future_cp[2],
        eval_after_4ply_cp=future_cp[4],
        eval_after_6ply_cp=future_cp[6],
        eval_after_10ply_cp=future_cp[10],
        win_after_2ply=future_win[2],
        win_after_4ply=future_win[4],
        win_after_6ply=future_win[6],
        win_after_10ply=future_win[10],
        horizon_2ply_reached=reached[2],
        horizon_4ply_reached=reached[4],
        horizon_6ply_reached=reached[6],
        horizon_10ply_reached=reached[10],
        game_ended_before_2ply=not reached[2],
        game_ended_before_4ply=not reached[4],
        game_ended_before_6ply=not reached[6],
        game_ended_before_10ply=not reached[10],
        terminal_result_used_for_2ply=terminal_used[2],
        terminal_result_used_for_4ply=terminal_used[4],
        terminal_result_used_for_6ply=terminal_used[6],
        terminal_result_used_for_10ply=terminal_used[10],
        mate_score_used_after=mate_after,
        mate_score_used_after_2ply=future_mate[2],
        mate_score_used_after_4ply=future_mate[4],
        mate_score_used_after_6ply=future_mate[6],
        mate_score_used_after_10ply=future_mate[10],
        initial_damage=initial_damage,
        remaining_damage_2ply=remaining[2],
        remaining_damage_4ply=remaining[4],
        remaining_damage_6ply=remaining[6],
        remaining_damage_10ply=remaining[10],
        persistence_ratio_2ply=ratios[2],
        persistence_ratio_4ply=ratios[4],
        persistence_ratio_6ply=ratios[6],
        persistence_ratio_10ply=ratios[10],
        persistence_ratio_2ply_clipped=clipped(ratios[2], config),
        persistence_ratio_4ply_clipped=clipped(ratios[4], config),
        persistence_ratio_6ply_clipped=clipped(ratios[6], config),
        persistence_ratio_10ply_clipped=clipped(ratios[10], config),
        persistence_category_2ply=classify_persistence(ratios[2], config),
        persistence_category_4ply=classify_persistence(ratios[4], config),
        persistence_category_6ply=classify_persistence(ratios[6], config),
        persistence_category_10ply=classify_persistence(ratios[10], config),
        another_large_mistake_within_2ply=any_mistakes[2],
        another_large_mistake_within_4ply=any_mistakes[4],
        another_large_mistake_within_6ply=any_mistakes[6],
        another_large_mistake_within_10ply=any_mistakes[10],
        original_player_large_mistake_within_2ply=original_mistakes[2],
        original_player_large_mistake_within_4ply=original_mistakes[4],
        original_player_large_mistake_within_6ply=original_mistakes[6],
        original_player_large_mistake_within_10ply=original_mistakes[10],
        opponent_large_mistake_within_2ply=opponent_mistakes[2],
        opponent_large_mistake_within_4ply=opponent_mistakes[4],
        opponent_large_mistake_within_6ply=opponent_mistakes[6],
        opponent_large_mistake_within_10ply=opponent_mistakes[10],
    )


def rows_for_game(
    game: chess.pgn.Game,
    game_index: int,
    source_pgn: str,
    engine: chess.engine.SimpleEngine,
    limit: chess.engine.Limit,
    config: Config,
) -> Iterable[DatasetRow]:
    records = collect_moves(game)
    board = game.board()

    for record in records:
        mover = board.turn
        board_before = board.copy(stack=False)
        before_info = engine.analyse(board, limit)
        eval_before, _ = cp_score_and_mate(before_info, mover, config.mate_score_cp)
        board.push(record.move)
        after_info = engine.analyse(board, limit)
        eval_after, mate_after = cp_score_and_mate(after_info, mover, config.mate_score_cp)
        cp_loss = eval_before - eval_after
        win_before = winning_chance(eval_before)
        win_drop = win_before - winning_chance(eval_after)

        if cp_loss < config.min_cp_loss:
            continue
        if win_drop < config.min_win_loss:
            continue
        if not (config.win_before_min <= win_before <= config.win_before_max):
            continue

        yield build_row(
            game=game,
            game_index=game_index,
            source_pgn=source_pgn,
            records=records,
            record=record,
            board=board_before,
            before_info=before_info,
            eval_before=eval_before,
            eval_after=eval_after,
            mate_after=mate_after,
            engine=engine,
            limit=limit,
            config=config,
        )


def write_rows(rows: Iterable[DatasetRow], output: Path) -> int:
    count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(DatasetRow)]

    if output.suffix.lower() == ".csv":
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))
                count += 1
        return count

    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def iter_rows(args: argparse.Namespace) -> Iterable[DatasetRow]:
    limit = limit_from_args(args)
    config = config_from_args(args)
    pgn_paths = [Path(path) for path in (args.pgn or ["lichess_db_standard_rated_2015-04.pgn.zst"])]
    game_index = 0
    analyzed_games = 0

    with chess.engine.SimpleEngine.popen_uci(args.engine) as engine:
        for pgn_path in pgn_paths:
            if analyzed_games >= args.max_games:
                break
            stream, zstd_proc = open_pgn_stream(pgn_path)
            try:
                while analyzed_games < args.max_games:
                    if game_index < args.start_game - 1:
                        if not chess.pgn.skip_game(stream):
                            break
                        game_index += 1
                        continue
                    game = chess.pgn.read_game(stream)
                    if game is None:
                        break
                    game_index += 1
                    analyzed_games += 1
                    yield from rows_for_game(
                        game, game_index, str(pgn_path), engine, limit, config
                    )
            finally:
                stream.close()
                if zstd_proc is not None:
                    zstd_proc.terminate()
                    zstd_proc.wait(timeout=5)


def run_self_tests() -> int:
    info = {"score": chess.engine.PovScore(chess.engine.Cp(123), chess.WHITE)}
    white_cp, white_mate = cp_score_and_mate(info, chess.WHITE, 1000)
    black_cp, black_mate = cp_score_and_mate(info, chess.BLACK, 1000)
    assert white_cp == 123
    assert black_cp == -123
    assert not white_mate and not black_mate

    mate_info = {"score": chess.engine.PovScore(chess.engine.Mate(3), chess.WHITE)}
    assert cp_score_and_mate(mate_info, chess.WHITE, 1000) == (1000, True)
    assert cp_score_and_mate(mate_info, chess.BLACK, 1000) == (-1000, True)
    assert winning_chance(100) > winning_chance(0) > winning_chance(-100)
    print("Self-tests passed: perspective normalization, mate conversion, win chance monotonicity.")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_tests()
    pgn_paths = [Path(path) for path in (args.pgn or ["lichess_db_standard_rated_2015-04.pgn.zst"])]
    missing_paths = [str(path) for path in pgn_paths if not path.exists()]
    if missing_paths:
        print(f"Input PGN not found: {', '.join(missing_paths)}", file=sys.stderr)
        return 2
    if not shutil.which(args.engine) and not Path(args.engine).exists():
        print("Stockfish executable was not found. Install it or pass --engine /path/to/stockfish.", file=sys.stderr)
        return 2

    count = write_rows(iter_rows(args), Path(args.output))
    print(f"Wrote {count} selected bad moves to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
