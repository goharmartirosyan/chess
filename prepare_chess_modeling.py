#!/usr/bin/env python3
"""Prepare the combined chess CSV for modeling, without altering the original.

Run: python3 prepare_chess_modeling.py --input combined_chess_games.csv

Outputs:
  combined_chess_games_modeling_gt20.csv  valid target at ALL four horizons
  combined_chess_games_excluded.csv       excluded rows with explicit reasons

IMPORTANT: PGN result cannot identify abandonment/disconnection by itself.
If trusted termination metadata is available, exclude unrelated terminations first.
"""
import argparse
import csv
import math
from collections import Counter
from pathlib import Path

HORIZONS = (2, 4, 6, 10)
RESULTS = {"1-0", "0-1", "1/2-1/2"}
DERIVED = ["change_in_damage_%sply" % h for h in HORIZONS]


def boolean(s):
    s = (s or "").strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    raise ValueError(f"Invalid boolean: {s!r}")


def number(s):
    s = (s or "").strip()
    if not s:
        return None
    n = float(s)
    if not math.isfinite(n):
        raise ValueError(f"Not a finite number: {s}")
    return n


def terminal_chance(result, side):
    if result == "1/2-1/2":
        return 50.0
    if result not in RESULTS:
        return None
    return 100.0 if ((result == "1-0") == (side == "White")) else 0.0


def category(ratio):
    if ratio < .25: return "mostly_recovered"
    if ratio < .75: return "partially_persistent"
    if ratio <= 1.0: return "strongly_persistent"
    return "worsened_further"


def prepare(row, threshold, require_all):
    """Return (corrected_row, exclusion_reasons)."""
    reasons = []
    before = number(row["win_before"])
    after = number(row["win_after"])
    if before is None or after is None:
        return row, ["missing_initial_win_chance"]
    if not (10 <= before <= 90):
        reasons.append("win_before_outside_10_90")
    damage = before - after  # recalculate rather than trust stale column
    if not damage > threshold:
        reasons.append("initial_damage_not_above_20")
    if reasons:
        return row, reasons

    side = row["side"].strip().capitalize()
    if side not in ("White", "Black"):
        return row, ["invalid_mistake_maker_side"]
    result = row["result"].strip()
    win_terminal = terminal_chance(result, side)
    row["initial_damage"] = repr(damage)

    for h in HORIZONS:
        suf = f"{h}ply"
        reached = boolean(row[f"horizon_{suf}_reached"])
        ended = boolean(row[f"game_ended_before_{suf}"])
        if reached == ended:  # exactly one must hold
            reasons.append(f"inconsistent_{suf}_flags")
            continue
        if ended:
            if win_terminal is None:
                reasons.append(f"unknown_terminal_result_{suf}")
                row[f"win_after_{suf}"] = ""
                row[f"terminal_result_used_for_{suf}"] = "False"
            else:
                row[f"win_after_{suf}"] = repr(win_terminal)
                row[f"terminal_result_used_for_{suf}"] = "True"
                row[f"eval_after_{suf}_cp"] = ""
                row[f"mate_score_used_after_{suf}"] = "False"
        else:
            row[f"terminal_result_used_for_{suf}"] = "False"

        future = number(row[f"win_after_{suf}"])
        if future is None:
            reasons.append(f"missing_{suf}_target")
            continue
        remaining = before - future
        ratio = remaining / damage
        row[f"remaining_damage_{suf}"] = repr(remaining)
        row[f"persistence_ratio_{suf}"] = repr(ratio)
        row[f"persistence_ratio_{suf}_clipped"] = repr(min(2.0, max(0.0, ratio)))
        row[f"persistence_category_{suf}"] = category(ratio)
        row[f"change_in_damage_{suf}"] = repr(remaining - damage)

    if not require_all:
        # A row can still be used for horizons with valid targets. Keep it
        # instead of excluding it from an all-horizon training file.
        reasons = [x for x in reasons if not (x.startswith("unknown_terminal_result_") or x.startswith("missing_") and x.endswith("_target"))]
    return row, reasons


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("combined_chess_games.csv"))
    p.add_argument("--threshold", type=float, default=20.0)
    args = p.parse_args()
    src = args.input.resolve()
    kept = src.with_name(src.stem + "_modeling_gt20.csv")
    excluded = src.with_name(src.stem + "_excluded.csv")
    for dst in (kept, excluded):
        if dst.exists():
            p.error(f"Refusing to overwrite existing output: {dst}")
    counts = Counter()
    try:
        with src.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
            if not cols or len(set(cols)) != len(cols):
                raise ValueError("Missing or duplicate CSV headers")
            needed = {"side", "result", "win_before", "win_after", "initial_damage"}
            for h in HORIZONS:
                s = f"{h}ply"
                needed.update({f"horizon_{s}_reached", f"game_ended_before_{s}",
                               f"win_after_{s}", f"terminal_result_used_for_{s}",
                               f"eval_after_{s}_cp", f"mate_score_used_after_{s}",
                               f"remaining_damage_{s}", f"persistence_ratio_{s}",
                               f"persistence_ratio_{s}_clipped", f"persistence_category_{s}"})
            if needed - set(cols):
                raise ValueError(f"Missing columns: {sorted(needed - set(cols))}")
            output_cols = list(cols) + [x for x in DERIVED if x not in cols]
            with kept.open("x", encoding="utf-8", newline="") as good, excluded.open("x", encoding="utf-8", newline="") as bad:
                wgood = csv.DictWriter(good, fieldnames=output_cols)
                wbad = csv.DictWriter(bad, fieldnames=output_cols + ["exclusion_reason"])
                wgood.writeheader(); wbad.writeheader()
                for row in reader:
                    counts["total"] += 1
                    if None in row:
                        raise ValueError(f"Malformed CSV at row {counts['total']}")
                    try:
                        changed, reasons = prepare(row.copy(), args.threshold, True)
                    except (ValueError, KeyError) as exc:
                        changed, reasons = row.copy(), ["invalid_data:" + str(exc)]
                    if reasons:
                        changed["exclusion_reason"] = ";".join(dict.fromkeys(reasons))
                        wbad.writerow(changed)
                        counts["excluded"] += 1
                        for r in set(reasons): counts[r] += 1
                    else:
                        wgood.writerow(changed)
                        counts["kept"] += 1
    except Exception:
        kept.unlink(missing_ok=True)
        excluded.unlink(missing_ok=True)
        raise
    print(f"Total rows: {counts['total']:,} | kept: {counts['kept']:,} | excluded: {counts['excluded']:,}")
    for key, val in counts.most_common():
        if key not in ("total", "kept", "excluded"):
            print(f"  {key}: {val:,}")
    print(f"Modeling CSV: {kept}\nExcluded CSV: {excluded}")
    print("Note: terminal reason cannot be inferred from PGN result alone; audit unrelated terminations separately.")

if __name__ == "__main__":
    main()
