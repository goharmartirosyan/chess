#!/usr/bin/env python3
"""Create a standalone HTML report for chess mistake-persistence relationships."""

from __future__ import annotations

import argparse
import csv
import html
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


TARGETS = (
    ("persistence_ratio_6ply", "Persistence ratio at 6 plies"),
    ("remaining_damage_6ply", "Remaining win-chance damage at 6 plies"),
)

# Initial damage is included for transparency, but it is mathematically coupled
# to both targets and should not be interpreted as an independent predictor.
NUMERIC_FEATURES = (
    ("player_elo", "Mistake-maker Elo"),
    ("opponent_elo", "Opponent Elo"),
    ("rating_difference", "Rating difference"),
    ("initial_time_seconds", "Initial clock (seconds)"),
    ("increment_seconds", "Increment (seconds)"),
    ("player_clock_before_move", "Mistake-maker clock (seconds)"),
    ("opponent_clock_before_move", "Opponent clock (seconds)"),
    ("clock_difference", "Clock difference (seconds)"),
    ("player_clock_pct_initial", "Mistake-maker clock (% initial)"),
    ("opponent_clock_pct_initial", "Opponent clock (% initial)"),
    ("eval_before_cp", "Evaluation before (cp)"),
    ("eval_after_cp", "Evaluation after (cp)"),
    ("centipawn_loss", "Centipawn loss"),
    ("win_before", "Win chance before"),
    ("win_after", "Win chance after"),
    ("win_chance_drop", "Win-chance drop / initial damage"),
    ("initial_damage", "Initial damage (mechanically related)"),
    ("move_number", "Move number"),
    ("material_balance_cp", "Material balance (cp)"),
    ("number_of_pieces", "Pieces on board"),
    ("number_of_pawns", "Pawns on board"),
    ("mobility", "Mistake-maker mobility"),
    ("opponent_mobility", "Opponent mobility"),
    ("legal_moves", "Legal moves / branching factor"),
    ("hanging_pieces", "Mistake-maker hanging pieces"),
    ("opponent_hanging_pieces", "Opponent hanging pieces"),
    ("king_attackers", "Attacks on mistake-maker king"),
    ("opponent_king_attackers", "Attacks on opponent king"),
    ("reasonable_candidate_moves", "Reasonable Stockfish candidates"),
    ("eval_gap_best_second_cp", "Best-to-second-best gap (cp)"),
    ("eval_gap_best_third_cp", "Best-to-third-best gap (cp)"),
)

CATEGORICAL_FEATURES = (
    ("game_phase", "Game phase"),
    ("time_control", "Time control"),
    ("side", "Mistake-maker side"),
    ("severity_label", "Classical severity label"),
    ("in_check", "In check before move"),
    ("gives_check", "Mistake gives check"),
    ("is_capture", "Mistake is a capture"),
    ("is_promotion", "Mistake is a promotion"),
    ("shallow_deep_best_agree", "Shallow/deep best move agree"),
    ("punishment_only_deep", "Best punishment only at deep search"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a standalone HTML persistence-feature analysis report."
    )
    parser.add_argument("--input", required=True, help="Extractor CSV to analyze.")
    parser.add_argument(
        "--output",
        default="feature_distributions.html",
        help="Standalone HTML report path (default: feature_distributions.html).",
    )
    parser.add_argument("--bins", type=int, default=5, help="Equal-frequency feature bins (default: 5).")
    parser.add_argument("--histogram-bins", type=int, default=20, help="Histogram bins (default: 20).")
    parser.add_argument("--min-group-size", type=int, default=20, help="Minimum categorical group size (default: 20).")
    parser.add_argument("--max-categories", type=int, default=15, help="Maximum groups per categorical feature (default: 15).")
    parser.add_argument(
        "--min-initial-damage",
        type=float,
        default=20.0,
        help="Analyze only rows with initial_damage strictly greater than this value (default: 20).",
    )
    return parser.parse_args()


def as_number(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summary(values: Iterable[float]) -> dict[str, float | int]:
    data = list(values)
    if not data:
        return {key: math.nan for key in ("min", "p05", "p25", "median", "mean", "p75", "p95", "max")} | {"n": 0}
    return {
        "n": len(data), "min": min(data), "p05": percentile(data, .05), "p25": percentile(data, .25),
        "median": statistics.median(data), "mean": statistics.fmean(data), "p75": percentile(data, .75),
        "p95": percentile(data, .95), "max": max(data),
    }


def pearson(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    x_mean, y_mean = statistics.fmean(xs), statistics.fmean(ys)
    x_scale = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_scale = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_scale == 0 or y_scale == 0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in pairs) / (x_scale * y_scale)


def average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        rank = (start + 1 + end) / 2
        for original_index, _ in ordered[start:end]:
            ranks[original_index] = rank
        start = end
    return ranks


def spearman(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    return pearson(list(zip(average_ranks(list(xs)), average_ranks(list(ys)))))


def quantile_groups(pairs: list[tuple[float, float]], count: int) -> list[tuple[str, list[float]]]:
    ordered = sorted(pairs)
    groups = []
    for index in range(min(count, len(ordered))):
        start = round(index * len(ordered) / min(count, len(ordered)))
        end = round((index + 1) * len(ordered) / min(count, len(ordered)))
        subset = ordered[start:end]
        if subset:
            groups.append((f"Q{index + 1}: {subset[0][0]:.2f} to {subset[-1][0]:.2f}", [value for _, value in subset]))
    return groups


def fmt(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return str(value) if isinstance(value, int) else f"{value:.3f}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    header_html = "".join(f"<th>{html.escape(cell)}</th>" for cell in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<div class=\"table-wrap\"><table><thead><tr>{header_html}</tr></thead><tbody>{body}</tbody></table></div>"


def histogram_svg(values: list[float], title: str, requested_bins: int) -> str:
    if not values:
        return "<p class=\"empty\">No values available.</p>"
    width, height, left, bottom = 680, 250, 48, 34
    low, high = min(values), max(values)
    if low == high:
        low, high = low - .5, high + .5
    bins = max(1, min(requested_bins, len(values)))
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, int((value - low) / (high - low) * bins))
        counts[index] += 1
    plot_width, plot_height = width - left - 12, height - bottom - 26
    max_count = max(counts)
    bar_width = plot_width / bins
    bars = []
    for index, count in enumerate(counts):
        bar_height = 0 if max_count == 0 else plot_height * count / max_count
        x, y = left + index * bar_width + 1, 18 + plot_height - bar_height
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(1, bar_width - 2):.1f}" height="{bar_height:.1f}" class="bar"><title>{count} rows</title></rect>')
    return f'''<svg viewBox="0 0 {width} {height}" role="img" aria-label="Histogram of {html.escape(title)}">
      <line x1="{left}" y1="{18 + plot_height}" x2="{width - 8}" y2="{18 + plot_height}" class="axis"/>
      <line x1="{left}" y1="18" x2="{left}" y2="{18 + plot_height}" class="axis"/>
      {''.join(bars)}
      <text x="{left}" y="{height - 10}" class="axis-text">{low:.2f}</text>
      <text x="{width - 8}" y="{height - 10}" text-anchor="end" class="axis-text">{high:.2f}</text>
      <text x="{left - 7}" y="22" text-anchor="end" class="axis-text">{max_count}</text>
    </svg>'''


def scatter_svg(pairs: list[tuple[float, float]], x_label: str, y_label: str) -> str:
    if len(pairs) < 2:
        return "<p class=\"empty\">Not enough complete rows for a chart.</p>"
    sample = pairs[::max(1, len(pairs) // 700)]
    width, height, left, bottom = 680, 290, 55, 45
    xs, ys = zip(*sample)
    x_low, x_high, y_low, y_high = min(xs), max(xs), min(ys), max(ys)
    if x_low == x_high:
        x_low, x_high = x_low - .5, x_high + .5
    if y_low == y_high:
        y_low, y_high = y_low - .5, y_high + .5
    plot_width, plot_height = width - left - 12, height - bottom - 16
    points = []
    for x, y in sample:
        px = left + (x - x_low) / (x_high - x_low) * plot_width
        py = 14 + (y_high - y) / (y_high - y_low) * plot_height
        points.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.3" class="point"/>')
    return f'''<svg viewBox="0 0 {width} {height}" role="img" aria-label="Scatter plot of {html.escape(y_label)} by {html.escape(x_label)}">
      <line x1="{left}" y1="{14 + plot_height}" x2="{width - 8}" y2="{14 + plot_height}" class="axis"/>
      <line x1="{left}" y1="14" x2="{left}" y2="{14 + plot_height}" class="axis"/>
      {''.join(points)}
      <text x="{left}" y="{height - 27}" class="axis-text">{x_low:.2f}</text>
      <text x="{width - 8}" y="{height - 27}" text-anchor="end" class="axis-text">{x_high:.2f}</text>
      <text x="{left - 5}" y="18" text-anchor="end" class="axis-text">{y_high:.2f}</text>
      <text x="{left - 5}" y="{14 + plot_height}" text-anchor="end" class="axis-text">{y_low:.2f}</text>
      <text x="{width / 2}" y="{height - 7}" text-anchor="middle" class="axis-text">{html.escape(x_label)}</text>
    </svg>'''


def correlation_svg(items: list[tuple[str, float]]) -> str:
    if not items:
        return "<p class=\"empty\">No correlations available.</p>"
    width, row_height, left, center = 680, 25, 230, 450
    height = len(items) * row_height + 34
    rows = []
    for index, (label, value) in enumerate(items):
        y = 20 + index * row_height
        x = center if value >= 0 else center + value * 190
        bar_width = abs(value) * 190
        color = "#16856a" if value >= 0 else "#b94a48"
        rows.append(f'<text x="{left - 8}" y="{y + 14}" text-anchor="end" class="axis-text">{html.escape(label)}</text>')
        rows.append(f'<rect x="{x:.1f}" y="{y}" width="{bar_width:.1f}" height="15" fill="{color}"><title>Spearman r = {value:.3f}</title></rect>')
    return f'''<svg viewBox="0 0 {width} {height}" role="img" aria-label="Spearman correlation bars">
      <line x1="{center - 190}" y1="10" x2="{center - 190}" y2="{height - 18}" class="grid"/>
      <line x1="{center}" y1="10" x2="{center}" y2="{height - 18}" class="axis"/>
      <line x1="{center + 190}" y1="10" x2="{center + 190}" y2="{height - 18}" class="grid"/>
      <text x="{center - 190}" y="{height - 3}" text-anchor="middle" class="axis-text">-1</text>
      <text x="{center}" y="{height - 3}" text-anchor="middle" class="axis-text">0</text>
      <text x="{center + 190}" y="{height - 3}" text-anchor="middle" class="axis-text">1</text>
      {''.join(rows)}
    </svg>'''


def grouped_stats(rows: list[dict[str, str]], target: str, feature: str, minimum: int, maximum: int) -> list[tuple[str, dict[str, float | int]]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = as_number(row.get(target))
        category = (row.get(feature) or "").strip()
        if value is not None and category:
            groups[category].append(value)
    selected = [(name, summary(values)) for name, values in groups.items() if len(values) >= minimum]
    return sorted(selected, key=lambda item: int(item[1]["n"]), reverse=True)[:maximum]


def html_document(body: str) -> str:
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chess Mistake Persistence Analysis</title>
<style>
  :root {{ color-scheme: light; --ink:#17212b; --muted:#5a6874; --line:#d8e0e5; --paper:#ffffff; --wash:#f4f7f8; --teal:#16856a; --red:#b94a48; }}
  body {{ margin:0; background:var(--wash); color:var(--ink); font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
  main {{ max-width:1240px; margin:0 auto; padding:28px 20px 56px; }} h1 {{ margin:0 0 4px; font-size:28px; }} h2 {{ margin:34px 0 12px; font-size:20px; }} h3 {{ margin:24px 0 10px; font-size:16px; }}
  p {{ color:var(--muted); }} .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:10px; margin:20px 0; }} .card {{ background:var(--paper); border:1px solid var(--line); border-radius:7px; padding:14px; }} .card b {{ display:block; font-size:22px; }} .card span {{ color:var(--muted); }}
  .chart-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:14px; }} .panel {{ background:var(--paper); border:1px solid var(--line); border-radius:7px; padding:14px; overflow:auto; }} .panel h3 {{ margin-top:0; }} svg {{ width:100%; height:auto; display:block; }} .axis {{ stroke:#8fa0aa; stroke-width:1; }} .grid {{ stroke:#d8e0e5; stroke-width:1; }} .axis-text {{ font-size:10px; fill:#5a6874; }} .bar {{ fill:#16856a; opacity:.86; }} .point {{ fill:#16856a; opacity:.5; }}
  table {{ width:100%; border-collapse:collapse; background:var(--paper); }} th,td {{ border-bottom:1px solid var(--line); padding:8px; text-align:right; white-space:nowrap; }} th:first-child,td:first-child {{ text-align:left; }} th {{ color:var(--muted); font-weight:600; background:#f8fafb; }} .table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:7px; }} .note {{ border-left:3px solid var(--teal); padding:10px 12px; background:#edf8f5; color:#2d4b45; }} .empty {{ color:var(--muted); }} code {{ background:#e9eef1; padding:1px 4px; border-radius:3px; }}
  @media (max-width:520px) {{ main {{ padding:20px 12px; }} .chart-grid {{ grid-template-columns:1fr; }} .panel {{ padding:10px; }} }}
</style></head><body><main>{body}</main></body></html>'''


def main() -> int:
    args = parse_args()
    input_path, output_path = Path(args.input), Path(args.output)
    if args.bins < 1 or args.histogram_bins < 1:
        raise SystemExit("--bins and --histogram-bins must be at least 1.")
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        raw_rows = list(reader)
    if "initial_damage" not in fieldnames:
        raise SystemExit("Input must contain initial_damage for --min-initial-damage filtering.")
    rows = [row for row in raw_rows if (damage := as_number(row.get("initial_damage"))) is not None and damage > args.min_initial_damage]
    available_targets = [(name, label) for name, label in TARGETS if name in fieldnames]
    if not available_targets:
        raise SystemExit("Input must contain persistence_ratio_6ply or remaining_damage_6ply.")

    body = [
        "<h1>Chess Mistake Persistence Analysis</h1>",
        f"<p>Source: <code>{html.escape(str(input_path))}</code></p>",
        "<div class=\"cards\">"
        f"<div class=\"card\"><b>{len(raw_rows):,}</b><span>Rows read</span></div>"
        f"<div class=\"card\"><b>{len(rows):,}</b><span>Rows analyzed</span></div>"
        f"<div class=\"card\"><b>&gt; {args.min_initial_damage:g}</b><span>Initial-damage filter</span></div>"
        "</div>",
        "<p class=\"note\">This report is descriptive, not causal. Numeric comparisons use only rows with both values present. A negative persistence ratio means the mistake-maker recovered beyond their pre-mistake win chance.</p>",
        "<h2>Persistence Target Distributions</h2>",
    ]

    target_table_rows = []
    target_values: dict[str, list[float]] = {}
    for target, label in available_targets:
        values = [value for row in rows if (value := as_number(row.get(target))) is not None]
        target_values[target] = values
        stats = summary(values)
        target_table_rows.append([label] + [fmt(stats[key]) for key in ("n", "min", "p05", "p25", "median", "mean", "p75", "p95", "max")])
    body.append(table(["Target", "N", "Min", "P05", "P25", "Median", "Mean", "P75", "P95", "Max"], target_table_rows))
    body.append("<div class=\"chart-grid\">" + "".join(
        f"<section class=\"panel\"><h3>{html.escape(label)}</h3>{histogram_svg(target_values[target], label, args.histogram_bins)}</section>"
        for target, label in available_targets
    ) + "</div>")

    for target, target_label in available_targets:
        associations = []
        pair_map: dict[str, list[tuple[float, float]]] = {}
        quantile_rows = []
        for feature, feature_label in NUMERIC_FEATURES:
            if feature not in fieldnames:
                associations.append((feature, feature_label, 0, None, None))
                continue
            pairs = [(feature_value, target_value) for row in rows if (feature_value := as_number(row.get(feature))) is not None and (target_value := as_number(row.get(target))) is not None]
            pair_map[feature] = pairs
            associations.append((feature, feature_label, len(pairs), pearson(pairs), spearman(pairs)))
            for group, values in quantile_groups(pairs, args.bins):
                stats = summary(values)
                quantile_rows.append([feature_label, group] + [fmt(stats[key]) for key in ("n", "median", "mean", "p25", "p75")])

        body.append(f"<h2>{html.escape(target_label)} Relationships</h2>")
        body.append("<p>Spearman is generally more robust for these non-linear, heavy-tailed chess targets. The initial-damage row is shown for transparency but is mechanically related to the target calculation.</p>")
        association_rows = [[label, str(n) if feature in fieldnames else "unavailable", fmt(p), fmt(s)] for feature, label, n, p, s in associations]
        body.append(table(["Numeric feature", "Complete rows", "Pearson r", "Spearman r"], association_rows))
        ranked = sorted([(label, spearman_value) for _, label, _, _, spearman_value in associations if spearman_value is not None], key=lambda item: abs(item[1]), reverse=True)[:12]
        body.append("<div class=\"chart-grid\"><section class=\"panel\"><h3>Strongest Spearman associations</h3>" + correlation_svg(ranked) + "</section>")
        top_features = [
            feature
            for feature, _, _, _, correlation in sorted(
                associations,
                key=lambda item: abs(item[4]) if item[4] is not None else -1,
                reverse=True,
            )
            if correlation is not None and pair_map.get(feature)
        ][:3]
        for feature in top_features:
            label = next(label for key, label in NUMERIC_FEATURES if key == feature)
            body.append(f"<section class=\"panel\"><h3>{html.escape(label)}</h3>{scatter_svg(pair_map[feature], label, target_label)}</section>")
        body.append("</div>")
        body.append("<h3>Target distribution by numeric-feature quintile</h3>")
        body.append(table(["Feature", "Feature range", "N", "Target median", "Target mean", "Target P25", "Target P75"], quantile_rows))

        category_rows = []
        for feature, label in CATEGORICAL_FEATURES:
            if feature not in fieldnames:
                category_rows.append([label, "unavailable", "-", "-", "-", "-", "-"])
                continue
            for group, stats in grouped_stats(rows, target, feature, args.min_group_size, args.max_categories):
                category_rows.append([label, group] + [fmt(stats[key]) for key in ("n", "median", "mean", "p25", "p75")])
        body.append("<h3>Target distribution by category</h3>")
        body.append(table(["Feature", "Group", "N", "Target median", "Target mean", "Target P25", "Target P75"], category_rows))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_document("\n".join(body)), encoding="utf-8")
    print(f"Wrote standalone HTML report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
